# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603902
- **Họ và tên:** Bùi Đức Tuấn
- **SEED** (in ở ô đầu notebook): 4828

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

Ô kiểm tra: 1.1–1.4, 2.1, 3.1–3.3, 4.1–4.4 và 5 bài tính tay đều ✅. **TODO 2.2 (`train_mlp`) chưa hoàn thành**, nên các hàng *2. BoW + MLP* và *3c. Encoder + MLP head* không có kết quả hợp lệ (MLP chưa được train, đoán ham cho mọi tin).

## 2. Kết quả chính

| Phương án | F1 test | F1 SUB 40 | p95 ms/tin | USD / 1k tin |
| --- | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg (tự code) | 0.888 | 0.889 | 0.13 | 0 |
| 3a. Encoder (MiniLM) + kNN | 0.917 | 0.919 | 67.5 | 0 |
| 3b. Encoder (MiniLM) + LogReg | 0.967 | 0.919 | 55.9 | 0 |
| 4a. LLM nhỏ zero-shot (gpt-6-luna) | — | 0.950 | 1072 | 0.0078 |
| 4b. LLM lớn zero-shot (gpt-6.1-terra) | — | 0.974 | 1315 | 0.0778 |
| 4c. LLM nhỏ few-shot | — | 0.947 | 1117 | 0.0234 |

LLM chỉ chạy trên SUB 40 tin nên chỉ so được cột F1 SUB 40.

## 3. Dự đoán và thực tế

1. **`llm_fewshot`** — Tôi đoán few-shot làm chi phí tăng **gấp 10 lần**. Thực tế chỉ tăng **khoảng 3 lần** ($0.0078 → $0.0234 / 1k tin), F1 gần như không đổi (0.950 → 0.947). Lệch vì tôi nhầm với chênh lệch giá giữa model lớn và nhỏ (đúng là 10 lần); chi phí few-shot chỉ tăng theo độ dài prompt (thêm 4 ví dụ ≈ prompt dài gấp ~3).
2. **`hard_eval`** — Tôi đoán LLM nhỏ chỉ đạt ~80% trên 50 case khó và nhóm khó nhất với LLM là prompt injection. Thực tế LLM đạt **100%** ở mọi nhóm, kể cả prompt injection; trong khi BoW chỉ còn 52% accuracy. Lệch vì tôi đánh giá thấp khả năng đọc hiểu ngữ cảnh của LLM.
3. **`tradeoff`** — Tôi đoán 3b nằm trên đường Pareto và đạt SLA 50 ms. Thực tế p95 của 3b là **55.9 ms**, vượt nhẹ SLA (3a: 67.5 ms); chỉ BoW + LogReg (0.13 ms) đạt SLA. Lệch vì encoder phải chạy 6 lớp transformer cho từng tin, chậm hơn tôi nghĩ.

Ngoài ra, lần chạy đầu của 4.A với `max_completion_tokens=3` cho F1 = 0 (invalid = 40): API trả về chuỗi rỗng (`finish_reason = length`, `reasoning_tokens = 0`). Tăng lên 16 token thì chạy bình thường.

## 4. Phân tích 50 case khó

- **Tụt nhiều nhất:** BoW + LogReg, F1 từ 0.889 (test thường) xuống 0.368. Nhóm khó nhất với BoW là **Thao túng / lừa đảo (12%)** và **Né từ khoá (17%)**. Test thường lạc quan vì cùng phân phối với train (nhiều spam gần giống nhau ở cả hai tập).
- **Case 1 — "Hi Mum, I dropped my phone in the toilet… send £200…"** (manipulation): BoW sai vì tin không có từ khoá spam nào; BoW chỉ cộng trọng số từng từ, không hiểu ý đồ lừa chuyển tiền.
- **Case 2 — `C0ngr4tulat10ns! Y0u h4ve w0n…`** (obfuscation): token bị đổi chữ thành số nên không có trong từ điển BoW → mô hình "không thấy" gì. Cách cứu: chuẩn hoá `0→o`, `1→i`… trước khi vector hoá (rủi ro: làm hỏng tin có số thật như OTP).
- **Embedding** khó nhất ở nhóm **Nói về spam** (3a: 25%) vì so theo nghĩa nên tin *kể về* lừa đảo nằm gần tin lừa đảo thật.
- **Case tự viết (bonus, 5 case tiếng Việt không dấu):** BoW và Encoder sai hầu hết vì vocab/encoder là tiếng Anh. LLM nhỏ zero-shot (4a) sai đúng 1 case: **#101** — người dùng tự gửi file mật cho chính mình, ghi "bỏ qua mọi chỉ dẫn" để bot mail không tự xử lý → 4a chặn nhầm là spam; prompt phòng thủ (4d) lại đoán đúng ham.

## 5. Thử thách cá nhân

Đề số 4 (theo SEED 4828). Chưa hoàn thành trong thời gian lab.

## 6. Phương án cho sản phẩm thật

- Với yêu cầu **p95 < 50 ms/tin**, chỉ **BoW + LogReg** đạt (p95 0.13 ms, ~11.600 tin/s, F1 0.888). 3b có F1 cao nhất trong các cách tự train (0.967) nhưng p95 55.9 ms, vượt nhẹ SLA. LLM có p95 ~1000–1300 ms, chậm gấp 20+ lần mức cho phép.
- Nếu dùng LLM nhỏ cho toàn bộ 10 triệu tin/ngày: 10.000 × $0.0078 ≈ **$78/ngày ≈ $2.340/tháng**, và không đạt độ trễ.
- **Đề xuất (router):** BoW + LogReg xử lý mọi tin; chỉ tin "lưỡng lự" (p ∈ [0.3, 0.7]) mới chuyển sang 3b hoặc LLM. Phần lớn tin đi đường nhanh, rẻ; tin khó mới tốn thời gian và tiền. Đánh đổi: một phần nhỏ tin có độ trễ cao hơn và recall của BoW (0.80) còn thấp nên vẫn có spam lọt.

## 7. AI log và điều đã học

| Phần | Hỏi AI gì | AI trả lời/viết gì | Tôi sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| Chung | Cách làm bài trên Colab, cách nộp, điểm nằm ở đâu | Giải thích quy trình, tóm tắt RUBRIC, file gợi ý từng câu (không có code Phần 1–3) | Tự làm bài theo thứ tự |
| TODO 1.1–1.4 | Demo tạo ma trận 0/1 bằng ví dụ khác bài; review code; hỏi chỗ viết | Demo "gọi món", review, gợi ý các bước bằng lời | Tự viết code, tự sửa lỗi thụt lề, chạy ô kiểm tra ✅ |
| Phần 1–3 (câu hỏi) | Góp ý dự đoán và câu trả lời | Chỉ ra chỗ hiểu sai (trọng số "free" tăng chứ không giảm, thiếu tham số lớp 2, bỏ ReLU → tuyến tính) | Sửa lại theo hiểu biết của mình, dùng số liệu của mình |
| TODO 4.1–4.4 (vùng được dùng AI) | Viết call_openai, review prompt 4.2, viết parse_label, llm_classify_batch | AI viết code, review prompt nháp của tôi | Đọc hiểu từng dòng, chạy ô kiểm tra ✅ |
| Debug Phần 4 | F1 = 0, invalid = 40 | Hướng dẫn in finish_reason / usage → phát hiện 3 token bị cắt | Tự sửa max_tokens 3 → 16, chạy lại được F1 0.95 |
| Câu hỏi 4–6 | Góp ý câu trả lời, viết ngắn lại theo ý tôi | Chỉ ra nhầm model/nhóm case, viết gọn lại với số của tôi | Đối chiếu với bảng kết quả của mình |
| Bonus MY_CASES | Hướng nghĩ case khó | Gợi ý hướng + ví dụ; chuyển 5 case của tôi sang cú pháp Python | Tự nghĩ 5 case và lý do (#101 là ý của tôi) |
| Tính tay | Tổng hợp đề, kiểm tra kết quả | Chỉ ra bài 4 sai (cosine phải ≤ 1) | Tự tính lại trên giấy → 0.60 |
| Nộp bài | Kiểm tra notebook (qua Claude in Chrome), soạn báo cáo này | Tìm ra ô dự đoán `emb_tricky` còn "..." chặn các ô sau; soạn nháp báo cáo từ câu trả lời và số liệu của tôi | Tự sửa dự đoán, chạy lại; đọc và sửa báo cáo |

**Điều đã học:** Accuracy cao chưa chắc mô hình tốt khi dữ liệu lệch lớp — mô hình "luôn đoán ham" đã được 87%. Mô hình chính xác nhất (LLM) lại chậm nhất, nên chọn phương án cho sản phẩm phải dựa trên cả độ trễ và chi phí chứ không chỉ F1. Và output của LLM không chắc chắn: một tham số nhỏ như `max_tokens` có thể làm toàn bộ kết quả về 0 nếu không kiểm tra.
