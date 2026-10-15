# BikeMind Dashboard

Dashboard Streamlit cho giám sát và điều phối mạng lưới xe đạp, gồm bốn phân hệ:

- Vận hành real-time với bản đồ Dark Matter và danh sách trạm rủi ro.
- Phân tích/dự báo nhu cầu 24 giờ và tải dữ liệu CSV.
- Đề xuất phân bổ, phê duyệt lệnh và bản đồ lộ trình.
- Cập nhật dữ liệu, retrain model và theo dõi MAE/RMSE/WAPE.

## Kiến trúc pipeline

Các module nền tảng được tách theo HLD:

- `pipeline/ingestion.py`: quét đệ quy CSV, loại file macOS rác, deduplicate theo MD5,
  loại bản ghi trùng và tạo lưới `station x hour` với zero-filling.
- `pipeline/forecasting.py`: tạo feature lịch, lag `1h/2h/24h/168h`, rolling `3h`,
  chronological split `70/15/15` và huấn luyện LightGBM nếu có; nếu chưa cài
  LightGBM, dùng HistGradientBoosting của scikit-learn với trạng thái fallback rõ ràng.
- `pipeline/rebalancing.py`: tính `Stock_expected`, phân loại deficit/surplus/balanced
  và ghép cặp trạm bằng khoảng cách Haversine greedy.
- `pipeline/service.py`: chạy end-to-end, train riêng pickups/returns, dự báo 24 giờ
  và nạp lại artifact cho dashboard.
- `pipeline/artifacts.py`: lưu Parquet, Pickle và `metrics.json`.
- `pipeline/benchmarks.py`: đo thời gian ingestion và rebalancing theo các tiêu chí HLD.

Pipeline có thể chạy độc lập để kiểm tra dữ liệu trước khi nối vào dashboard:

```python
from pipeline.ingestion import load_trip_data, build_station_hour_grid
from pipeline.forecasting import make_features, train_forecaster

trips, report = load_trip_data("data")
grid, stations = build_station_hour_grid(trips)
features = make_features(grid, target="pickups")
model, metrics = train_forecaster(features)
print(report, model.backend, metrics)
```

## Train và artifact

Chạy từ thư mục `main` để tạo dữ liệu thật cho dashboard:

```powershell
cd D:\KLTN\BikeMind\main
python -c "from pipeline.service import train_and_persist; train_and_persist('data', 'artifacts')"
```

Các file được tạo trong `artifacts/`:

- `station_metadata.parquet`: metadata của toàn bộ station xuất hiện ở đầu hoặc cuối chuyến.
- `station_hour_cleaned.parquet`: lưới station × hour đã zero-fill.
- `lgb_pickups.pkl`, `lgb_returns.pkl`: hai model dự báo độc lập (LightGBM hoặc fallback có ghi rõ backend).
- `metrics.json`: MAE, RMSE, WAPE, thống kê ingestion và backend model.
- `dashboard_snapshot.json`: snapshot nhẹ để dashboard mở nhanh mà không phải đọc lại
  toàn bộ grid Parquet 169 MB ở mỗi lần render.

Dashboard bắt buộc đọc artifact thật; nếu artifact thiếu, ứng dụng dừng và yêu cầu
chạy retrain thay vì hiển thị dữ liệu demo. Snapshot hiển thị station, tọa độ, nhu cầu
24 giờ gần nhất và dự báo t+1 từ dữ liệu thật. Dataset chuyến đi không có trạng thái
tồn kho/capacity real-time, nên hai trường này được đánh dấu là ước tính từ lưu lượng
quan sát, không phải số liệu vận hành trực tiếp. Nút **Retrain Model** thực hiện lại
toàn bộ luồng, bao gồm file CSV upload.

## Chạy local

```powershell
cd D:\KLTN\BikeMind\main
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Bản đồ sử dụng nền CartoDB (Light/Dark theo toggle) thông qua Plotly, không cần API key.
