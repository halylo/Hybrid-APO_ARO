# 🚀 Hybrid APO–ARO Optimization & Benchmark Comparison

## 📌 Giới thiệu

Project này thực hiện **so sánh hiệu năng các thuật toán tối ưu hóa metaheuristic**, bao gồm:

* APO (Artificial Protozoa Optimization)
* ARO (Artificial Rabbits Optimization)
* AOA (Arithmetic Optimization Algorithm)
* COA (Coyote Optimization Algorithm)
* EFO (Electromagnetic Field Optimization)
* PSO (Particle Swarm Optimization)
* 🔥 Hybrid ARO–APO (đề xuất)

Mục tiêu:
👉 Đánh giá khả năng hội tụ và tối ưu trên các hàm benchmark chuẩn.

---

## 🧠 Các hàm benchmark

Project sử dụng 6 hàm:

* Sphere
* Schwefel 2.22
* Max Absolute
* Generalized Power
* Composite Quadratic
* Ackley

Các hàm này được định nghĩa trong:
👉 `benchmark_func.py` 

---

## ⚙️ Cấu hình

```python
N = 50        # Số cá thể
T = 1000      # Số vòng lặp
dim = 70      # Số chiều
```

Định nghĩa trong:
👉 `benchmark.py` 

---

## 🧪 Thuật toán triển khai

Các thuật toán chính nằm trong:

👉 `combine_apo_aro.py` 

Bao gồm:

* ARO
* APO
* AOA (PA1)
* COA (PA2)
* EFO (PA3)
* 🔥 ARO_APO (Hybrid)
* PSO (PA5)

---

## 🔬 Cách chạy

### 1. Cài thư viện

```bash
pip install numpy pandas matplotlib scipy
```

---

### 2. Chạy benchmark

```bash
python combine_apo_aro.py
```

👉 Kết quả sẽ tạo ra các file:

* `apo.csv`
* `aro.csv`
* `PA1_aoa.csv`
* `PA2_coa.csv`
* `PA3_efo.csv`
* `PA4_aro_apo.csv`
* `PA5_pso.csv`

---

### 3. Vẽ biểu đồ so sánh

```bash
python visualization.py
```

👉 File kết quả:

* `comparison_all.png`

Script tại:
👉 `visualization.py` 

---

## 📊 Kết quả

* So sánh tốc độ hội tụ của các thuật toán
* Hiển thị dưới dạng đồ thị log-scale
* Đánh giá hiệu quả thuật toán hybrid

---

## 📂 Cấu trúc project

```bash
.
├── benchmark.py
├── benchmark_func.py
├── combine_apo_aro.py
├── visualization.py
├── apo.csv
├── aro.csv
├── PA1_aoa.csv
├── PA2_coa.csv
├── PA3_efo.csv
├── PA4_aro_apo.csv
├── PA5_pso.csv
└── comparison_all.png
```

---

## 💡 Ý tưởng chính

* Kết hợp ARO và APO theo vòng lặp
* Tận dụng:

  * ARO → khai thác (exploitation)
  * APO → khám phá (exploration)

👉 Tăng khả năng tìm nghiệm tối ưu toàn cục

---

## ⚠️ Lưu ý

* Kết quả phụ thuộc random seed
* Dim lớn (70) → thời gian chạy lâu
* File CSV sẽ bị ghi đè mỗi lần chạy

---

## 👨‍💻 Research Team

This research is conducted by a group of students from  
**Ho Chi Minh City University of Industry and Trade (HUIT)**.

---

## ⭐ Hướng phát triển

* Thêm nhiều benchmark (CEC, IEEE)
* So sánh statistical (mean, std, Wilcoxon test)
* Parallel computing
* GUI visualization

