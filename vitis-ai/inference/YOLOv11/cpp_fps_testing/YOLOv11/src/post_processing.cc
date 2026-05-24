#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <dirent.h>
#include <sys/stat.h>

#include <opencv2/opencv.hpp>

// Vitis AI OS (PetaLinux 2022.2): build with build_post_processing_vitis_ai.sh
// Ubuntu/host: build with build_post_processing.sh. No glog/VART.

namespace {

const float kEps = 1e-7f;
const float kConfThres = 0.001f;
const int kMaxDet = 300;

// ---------------------------------------------------------------------------
// Load meta.txt: num_images, num_outputs, shapes per output, fixpoints
// ---------------------------------------------------------------------------
struct Meta {
  int num_images;
  int num_outputs;
  std::vector<std::vector<int>> shapes;  // per output: e.g. {1, 52, 52, 65}
  std::vector<int> fixpoints;
};

bool LoadMeta(const std::string& output_dir, Meta* meta) {
  std::ifstream f(output_dir + "/meta.txt");
  if (!f) return false;
  f >> meta->num_images >> meta->num_outputs;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    std::istringstream ss(line);
    int v;
    std::vector<int> shape;
    while (ss >> v) shape.push_back(v);
    if (!shape.empty()) {
      if (meta->shapes.size() < static_cast<size_t>(meta->num_outputs))
        meta->shapes.push_back(shape);
      else {
        meta->fixpoints = shape;
        break;
      }
    }
  }
  if (meta->shapes.size() != static_cast<size_t>(meta->num_outputs)) return false;
  if (meta->fixpoints.size() != static_cast<size_t>(meta->num_outputs)) {
    meta->fixpoints.clear();
    for (int i = 0; i < meta->num_outputs; ++i) { int x; if (!(f >> x)) return false; meta->fixpoints.push_back(x); }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Load one .bin file: int8, reshape to shape, dequantize to float
// ---------------------------------------------------------------------------
bool LoadBin(const std::string& path, const std::vector<int>& shape,
             int fixpoint, std::vector<float>* out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  f.seekg(0, std::ios::end);
  size_t sz = f.tellg();
  f.seekg(0);
  size_t n = 1;
  for (int s : shape) n *= s;
  if (sz != n * sizeof(int8_t)) return false;
  std::vector<int8_t> buf(n);
  f.read(reinterpret_cast<char*>(buf.data()), n);
  if (!f) return false;
  float scale = 1.f / (1 << fixpoint);
  out->resize(n);
  for (size_t i = 0; i < n; ++i) (*out)[i] = buf[i] * scale;
  return true;
}

// NHWC (1,H,W,C) -> NCHW (1,C,H,W), row-major
void NHWCtoNCHW(const std::vector<float>& in, const std::vector<int>& shape,
                std::vector<float>* out) {
  int N = shape[0], H = shape[1], W = shape[2], C = shape[3];
  out->resize(N * C * H * W);
  for (int n = 0; n < N; ++n)
    for (int c = 0; c < C; ++c)
      for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w)
          (*out)[n * C * H * W + c * H * W + h * W + w] =
              in[n * H * W * C + h * W * C + w * C + c];
}

// ---------------------------------------------------------------------------
// DFL: [B, 4*reg_max, A] -> softmax on reg_max dim -> [B, 4, A]
// ---------------------------------------------------------------------------
void ApplyDFL(std::vector<float>* box_preds, int B, int reg_max, int A) {
  std::vector<float> out(B * 4 * A);
  for (int b = 0; b < B; ++b) {
    for (int k = 0; k < 4; ++k) {
      for (int a = 0; a < A; ++a) {
        float max_v = -1e9f;
        for (int r = 0; r < reg_max; ++r) {
          float v = (*box_preds)[b * (4 * reg_max) * A + (k * reg_max + r) * A + a];
          if (v > max_v) max_v = v;
        }
        float sum = 0.f;
        for (int r = 0; r < reg_max; ++r) {
          float v = (*box_preds)[b * (4 * reg_max) * A + (k * reg_max + r) * A + a];
          sum += std::exp(v - max_v);
        }
        float weighted = 0.f;
        for (int r = 0; r < reg_max; ++r) {
          float v = (*box_preds)[b * (4 * reg_max) * A + (k * reg_max + r) * A + a];
          weighted += r * (std::exp(v - max_v) / sum);
        }
        out[b * 4 * A + k * A + a] = weighted;
      }
    }
  }
  *box_preds = std::move(out);
}

// ---------------------------------------------------------------------------
// dist2bbox: distance (1, 4, A), anchor_points (2, A) -> (1, 4, A) xywh
// ---------------------------------------------------------------------------
void Dist2Bbox(const std::vector<float>& distance,
               const std::vector<float>& anchor_points, int A,
               std::vector<float>* out) {
  out->resize(4 * A);
  for (int a = 0; a < A; ++a) {
    float ax = anchor_points[0 * A + a];
    float ay = anchor_points[1 * A + a];
    float lt0 = distance[0 * A + a];
    float lt1 = distance[1 * A + a];
    float rb0 = distance[2 * A + a];
    float rb1 = distance[3 * A + a];
    float x1 = ax - lt0, y1 = ay - lt1, x2 = ax + rb0, y2 = ay + rb1;
    (*out)[0 * A + a] = (x1 + x2) * 0.5f;
    (*out)[1 * A + a] = (y1 + y2) * 0.5f;
    (*out)[2 * A + a] = x2 - x1;
    (*out)[3 * A + a] = y2 - y1;
  }
}

// ---------------------------------------------------------------------------
// Sigmoid
// ---------------------------------------------------------------------------
inline float Sigmoid(float x) {
  return 1.f / (1.f + std::exp(-x));
}

// ---------------------------------------------------------------------------
// IoU (xyxy)
// ---------------------------------------------------------------------------
float BoxIou(const float* a, const float* b) {
  float ax1 = a[0], ay1 = a[1], ax2 = a[2], ay2 = a[3];
  float bx1 = b[0], by1 = b[1], bx2 = b[2], by2 = b[3];
  float ix1 = std::max(ax1, bx1), iy1 = std::max(ay1, by1);
  float ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);
  float iw = std::max(0.f, ix2 - ix1), ih = std::max(0.f, iy2 - iy1);
  float inter = iw * ih;
  float area_a = (ax2 - ax1) * (ay2 - ay1);
  float area_b = (bx2 - bx1) * (by2 - by1);
  float u = area_a + area_b - inter + kEps;
  return inter / u;
}

// xywh to xyxy
void Xywh2Xyxy(float x, float y, float w, float h, float* x1, float* y1, float* x2, float* y2) {
  *x1 = x - w * 0.5f;
  *y1 = y - h * 0.5f;
  *x2 = x + w * 0.5f;
  *y2 = y + h * 0.5f;
}

// ---------------------------------------------------------------------------
// NMS: prediction (1, 5, A) -> rows [x, y, w, h, conf]; filter by conf, then NMS
// Returns list of (x1, y1, w, h, conf) for output file
// ---------------------------------------------------------------------------
void NMS(const std::vector<float>& pred, int A, float conf_thres, float iou_thres,
         std::vector<std::vector<float>>* boxes_out) {
  struct Det {
    float x1, y1, x2, y2, conf;
    int idx;
  };
  std::vector<Det> dets;
  for (int a = 0; a < A; ++a) {
    float conf = pred[4 * A + a];
    if (conf <= conf_thres) continue;
    float x = pred[0 * A + a], y = pred[1 * A + a];
    float w = pred[2 * A + a], h = pred[3 * A + a];
    float x1, y1, x2, y2;
    Xywh2Xyxy(x, y, w, h, &x1, &y1, &x2, &y2);
    dets.push_back({x1, y1, x2, y2, conf, a});
  }
  std::sort(dets.begin(), dets.end(),
            [](const Det& a, const Det& b) { return a.conf > b.conf; });
  if (dets.size() > static_cast<size_t>(kMaxDet))
    dets.resize(kMaxDet);

  std::vector<int> keep;
  for (size_t i = 0; i < dets.size(); ++i) {
    bool suppress = false;
    for (int j : keep) {
      float iou = BoxIou(&dets[i].x1, &dets[j].x1);
      if (iou > iou_thres) { suppress = true; break; }
    }
    if (!suppress) keep.push_back(static_cast<int>(i));
  }

  boxes_out->clear();
  for (int i : keep) {
    const Det& d = dets[i];
    float w = d.x2 - d.x1, h = d.y2 - d.y1;
    boxes_out->push_back({d.x1, d.y1, w, h, d.conf});
  }
}

void MakeAnchorsFromShapes(const std::vector<std::vector<int>>& shapes,
                           const std::vector<int>& strides, float grid_cell_offset,
                           std::vector<float>* anchor_points,
                           std::vector<float>* stride_tensor) {
  std::vector<float> ap, st;
  for (size_t i = 0; i < shapes.size(); ++i) {
    int H = shapes[i][1], W = shapes[i][2];
    int s = strides[i];
    for (int h = 0; h < H; ++h)
      for (int w = 0; w < W; ++w) {
        ap.push_back(w + grid_cell_offset);
        ap.push_back(h + grid_cell_offset);
        st.push_back(static_cast<float>(s));
      }
  }
  size_t A = ap.size() / 2;
  anchor_points->resize(2 * A);
  for (size_t i = 0; i < A; ++i) {
    (*anchor_points)[0 * A + i] = ap[i * 2 + 0];
    (*anchor_points)[1 * A + i] = ap[i * 2 + 1];
  }
  stride_tensor->resize(A);
  for (size_t i = 0; i < A; ++i) (*stride_tensor)[i] = st[i];
}

bool ProcessOneImageImpl(const std::string& output_dir, int img_idx,
                         const Meta& meta, int reg_max, int num_classes,
                         const std::vector<int>& strides, float iou_thres,
                         std::vector<std::vector<float>>* boxes_out) {
  const int num_outputs = meta.num_outputs;
  if (num_outputs != 3 || static_cast<int>(strides.size()) != 3) return false;

  std::vector<std::vector<float>> feats_nchw;
  int total_A = 0;
  std::vector<int> heights, widths;
  for (int o = 0; o < num_outputs; ++o) {
    std::string path = output_dir + "/pred_" + std::to_string(img_idx) +
                       "_output_" + std::to_string(o) + ".bin";
    std::vector<float> dequant;
    if (!LoadBin(path, meta.shapes[o], meta.fixpoints[o], &dequant)) return false;
    std::vector<float> nchw;
    NHWCtoNCHW(dequant, meta.shapes[o], &nchw);
    feats_nchw.push_back(std::move(nchw));
    int H = meta.shapes[o][1], W = meta.shapes[o][2];
    heights.push_back(H);
    widths.push_back(W);
    total_A += H * W;
  }

  int C = meta.shapes[0][3];
  int reg_x_4 = reg_max * 4;
  if (C != reg_x_4 + num_classes) return false;

  std::vector<float> x_cat(1 * C * total_A, 0.f);
  int offset = 0;
  for (int o = 0; o < num_outputs; ++o) {
    int H = heights[o], W = widths[o];
    int count = H * W;
    const std::vector<float>& f = feats_nchw[o];
    for (int c = 0; c < C; ++c)
      for (int a = 0; a < count; ++a)
        x_cat[c * total_A + offset + a] = f[c * count + a];
    offset += count;
  }

  std::vector<float> box_vec(1 * reg_x_4 * total_A);
  for (int i = 0; i < reg_x_4 * total_A; ++i) box_vec[i] = x_cat[i];
  std::vector<float> cls_vec(total_A);
  for (int a = 0; a < total_A; ++a) cls_vec[a] = x_cat[reg_x_4 * total_A + a];

  ApplyDFL(&box_vec, 1, reg_max, total_A);

  std::vector<float> anchor_points, stride_tensor;
  MakeAnchorsFromShapes(meta.shapes, strides, 0.5f, &anchor_points, &stride_tensor);

  std::vector<float> dbox(4 * total_A);
  Dist2Bbox(box_vec, anchor_points, total_A, &dbox);
  for (int a = 0; a < total_A; ++a) {
    dbox[0 * total_A + a] *= stride_tensor[a];
    dbox[1 * total_A + a] *= stride_tensor[a];
    dbox[2 * total_A + a] *= stride_tensor[a];
    dbox[3 * total_A + a] *= stride_tensor[a];
  }

  std::vector<float> pred(5 * total_A);
  for (int a = 0; a < total_A; ++a) {
    pred[0 * total_A + a] = dbox[0 * total_A + a];
    pred[1 * total_A + a] = dbox[1 * total_A + a];
    pred[2 * total_A + a] = dbox[2 * total_A + a];
    pred[3 * total_A + a] = dbox[3 * total_A + a];
    pred[4 * total_A + a] = Sigmoid(cls_vec[a]);
  }

  NMS(pred, total_A, kConfThres, iou_thres, boxes_out);
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Config: reg_max nc stride0 stride1 stride2 (one line file or default)
// ---------------------------------------------------------------------------
bool LoadConfig(const std::string& path, int* reg_max, int* nc,
                std::vector<int>* strides) {
  std::ifstream f(path);
  if (!f) return false;
  f >> *reg_max >> *nc;
  int s;
  while (f >> s) strides->push_back(s);
  return strides->size() == 3;
}

void DefaultConfig(int* reg_max, int* nc, std::vector<int>* strides) {
  *reg_max = 16;
  *nc = 1;
  strides->assign({8, 16, 32});
}

// ---------------------------------------------------------------------------
// List image paths (same order as C++ inference)
// ---------------------------------------------------------------------------
std::vector<std::string> GetImagePaths(const std::string& dir) {
  std::vector<std::string> paths;
  DIR* d = opendir(dir.c_str());
  if (!d) return paths;
  struct dirent* ent;
  while ((ent = readdir(d)) != nullptr) {
    std::string name = ent->d_name;
    if (name == "." || name == "..") continue;
    size_t dot = name.rfind('.');
    if (dot == std::string::npos) continue;
    std::string ext = name.substr(dot);
    for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
      paths.push_back(dir + "/" + name);
  }
  closedir(d);
  std::sort(paths.begin(), paths.end());
  return paths;
}

// ---------------------------------------------------------------------------
// Draw boxes on image and save
// ---------------------------------------------------------------------------
void DrawAndSave(const std::string& img_path, const std::string& out_path,
                 const std::vector<std::vector<float>>& boxes, int img_size) {
  cv::Mat img = cv::imread(img_path);
  if (img.empty()) return;
  int W = img.cols, H = img.rows;
  float scale_x = static_cast<float>(W) / img_size;
  float scale_y = static_cast<float>(H) / img_size;
  for (const auto& b : boxes) {
    if (b.size() < 5) continue;
    int x = static_cast<int>(b[0] * scale_x);
    int y = static_cast<int>(b[1] * scale_y);
    int w = static_cast<int>(b[2] * scale_x);
    int h = static_cast<int>(b[3] * scale_y);
    cv::rectangle(img, cv::Point(x, y), cv::Point(x + w, y + h), cv::Scalar(0, 255, 0), 2);
    char buf[32];
    snprintf(buf, sizeof(buf), "%.2f", b[4]);
    std::string label = std::string("person ") + buf;
    int baseline = 0;
    cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    cv::rectangle(img, cv::Point(x, y - ts.height - 5), cv::Point(x + ts.width, y),
                  cv::Scalar(0, 255, 0), -1);
    cv::putText(img, label, cv::Point(x, y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 0, 0), 1);
  }
  cv::imwrite(out_path, img);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  if (argc < 6) {
    std::cerr << "Usage: " << argv[0]
              << " <output_dir> <config_file|-> <iou_threshold> <output_boxes.txt> [images_dir] [save_annotated_dir] [img_size]\n"
              << "  output_dir: directory with meta.txt and pred_*_output_*.bin\n"
              << "  config_file: text file with one line 'reg_max num_classes stride0 stride1 stride2' (e.g. 16 1 8 16 32). Use '-' for default.\n"
              << "  iou_threshold: NMS IoU (e.g. 0.5)\n"
              << "  output_boxes.txt: output file for boxes\n"
              << "  Optional: images_dir save_annotated_dir img_size (e.g. data rpt/annotated 416)\n";
    return 1;
  }

  std::string output_dir = argv[1];
  std::string config_arg = argv[2];
  float iou_thres = std::stof(argv[3]);
  std::string output_file = argv[4];
  std::string images_dir, save_annotated_dir;
  int img_size = 416;
  if (argc >= 8) {
    images_dir = argv[5];
    save_annotated_dir = argv[6];
    img_size = std::stoi(argv[7]);
  }

  Meta meta;
  if (!LoadMeta(output_dir, &meta)) {
    std::cerr << "Failed to load meta.txt from " << output_dir << "\n";
    return 1;
  }

  int reg_max, nc;
  std::vector<int> strides;
  if (config_arg == "-" || config_arg == "") {
    DefaultConfig(&reg_max, &nc, &strides);
  } else {
    if (!LoadConfig(config_arg, &reg_max, &nc, &strides)) {
      std::cerr << "Failed to load config from " << config_arg << "\n";
      return 1;
    }
  }

  std::ofstream out(output_file);
  if (!out) {
    std::cerr << "Cannot write " << output_file << "\n";
    return 1;
  }

  std::vector<std::string> image_paths;
  if (!images_dir.empty() && !save_annotated_dir.empty()) {
    image_paths = GetImagePaths(images_dir);
    mkdir(save_annotated_dir.c_str(), 0755);
  }

  for (int img_idx = 0; img_idx < meta.num_images; ++img_idx) {
    std::vector<std::vector<float>> boxes;
    if (!ProcessOneImageImpl(output_dir, img_idx, meta, reg_max, nc, strides,
                            iou_thres, &boxes)) {
      std::cerr << "ProcessOneImage failed for image " << img_idx << "\n";
      return 1;
    }

    out << "Image Prediction: image_" << img_idx << "\n\nBoxes:\n\n";
    for (const auto& b : boxes) {
      if (b.size() >= 5)
        out << b[0] << " " << b[1] << " " << b[2] << " " << b[3] << " " << b[4] << "\n";
    }
    out << "\n";

    if (img_idx < static_cast<int>(image_paths.size())) {
      std::string out_path = save_annotated_dir + "/det_" +
          image_paths[img_idx].substr(image_paths[img_idx].rfind('/') + 1);
      DrawAndSave(image_paths[img_idx], out_path, boxes, img_size);
      std::cout << "  Saved annotated: " << out_path << "\n";
    }
  }

  std::cout << "Results saved to " << output_file << "\n";
  return 0;
}
