import streamlit as st
import numpy as np
from datetime import datetime
import pytz
from pytz import utc
from suncalc import get_position
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

st.set_page_config(page_title="Определение локации по тени", layout="wide")

def compute_map_and_top(object_height, shadow_length, date_time_utc,
lat_min=-60, lat_max=85, lon_min=-180, lon_max=180,
lat_step=1.0, lon_step=1.0, threshold=0.15, top_n=5):

Сетка
lats = np.arange(lat_min, lat_max + 1e-9, lat_step)
lons = np.arange(lon_min, lon_max + 1e-9, lon_step)
lons_grid, lats_grid = np.meshgrid(lons, lats)

Высота солнца
sun_altitudes = np.zeros_like(lons_grid, dtype=float)
for i in range(len(lats)):
for j in range(len(lons)):
pos = get_position(date_time_utc, float(lons[j]), float(lats[i]))
sun_altitudes[i, j] = pos["altitude"] # radians

Теоретическая длина тени
with np.errstate(divide="ignore", invalid="ignore"):
calc_shadows = object_height / np.tan(sun_altitudes)

Относительная разница
differences = np.abs((calc_shadows - shadow_length) / shadow_length)

Маскируем невозможные точки
differences[calc_shadows < 0] = np.nan
differences[sun_altitudes <= 0] = np.nan

Готовим картинку
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
ax.add_feature(cfeature.OCEAN, alpha=0.3)
ax.add_feature(cfeature.LAND, alpha=0.3)
ax.add_feature(cfeature.LAKES, alpha=0.3)
ax.add_feature(cfeature.RIVERS, linewidth=0.5)

masked = np.ma.masked_where((differences > threshold) | np.isnan(differences), differences)
im = plt.pcolormesh(lons_grid, lats_grid, masked,
transform=ccrs.PlateCarree(), cmap="viridis", alpha=0.8, shading="auto")
cbar = plt.colorbar(im, orientation="horizontal", pad=0.05, aspect=50)
cbar.set_label("Относительная разница (меньше — лучше)")

title = (f"Возможные локации на {date_time_utc.strftime('%Y-%m-%d %H:%M UTC')}\n"
f"Высота объекта: {object_height}, Длина тени: {shadow_length}")
plt.title(title)

gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
gl.top_labels = False
gl.right_labels = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

Топ-N
flat_diffs = differences.ravel()
flat_lats = lats_grid.ravel()
flat_lons = lons_grid.ravel()
valid = ~np.isnan(flat_diffs)
if valid.sum() == 0:
top_points = []
else:
valid_idx = np.where(valid)[0]
sorted_idx = np.argsort(flat_diffs[valid_idx])
take = min(top_n, sorted_idx.size)
chosen = valid_idx[sorted_idx[:take]]
top_points = [{"lat": float(flat_lats[i]), "lon": float(flat_lons[i]), "diff": float(flat_diffs[i])}
for i in chosen]

return fig, top_points

UI
st.title("Определение возможной локации по высоте объекта и длине тени")
st.caption("Загрузи фото (для наглядности), введи параметры и получи карту с топ-координатами. "
"Время обязательно указывай с учетом часового пояса.")

with st.sidebar:
st.header("Ввод данных")
uploaded = st.file_uploader("Фото (опционально)", type=["jpg", "jpeg", "png"])
object_height = st.number_input("Высота объекта (в метрах или тех же единицах, что и тень)", min_value=0.1, value=98.0, step=0.5)
shadow_length = st.number_input("Длина тени (в тех же единицах)", min_value=0.1, value=185.0, step=0.5)

st.markdown("Дата и время съемки")
col1, col2 = st.columns(2)
with col1:
date_val = st.date_input("Дата", value=datetime.utcnow().date())
with col2:
time_val = st.time_input("Время", value=datetime.utcnow().time().replace(second=0, microsecond=0))

tz_list = ["UTC"] + pytz.all_timezones
tz_name = st.selectbox("Часовой пояс", options=tz_list, index=tz_list.index("UTC"))

st.markdown("Параметры карты (можно оставить по умолчанию)")
lat_step = st.select_slider("Шаг широты, ° (меньше — точнее, но медленнее)", options=[0.5, 1.0, 2.0], value=1.0)
lon_step = st.select_slider("Шаг долготы, °", options=[0.5, 1.0, 2.0], value=1.0)
threshold = st.select_slider("Порог разницы", options=[0.05, 0.1, 0.15, 0.2], value=0.15)

calc_btn = st.button("Рассчитать", type="primary")

Показ фото
if uploaded:
st.subheader("Загруженное фото")
st.image(uploaded, use_column_width=True)

Логика расчета
if calc_btn:
try:

Собираем локальное время и конвертируем в UTC
naive = datetime.combine(date_val, time_val)
local_tz = pytz.timezone(tz_name)
local_dt = local_tz.localize(naive)
dt_utc = local_dt.astimezone(utc)

with st.spinner("Считаем карту... Это может занять 10–60 секунд."):
fig, top_points = compute_map_and_top(
object_height=object_height,
shadow_length=shadow_length,
date_time_utc=dt_utc,
lat_step=float(lat_step),
lon_step=float(lon_step),
threshold=float(threshold),
top_n=5
)

st.subheader("Карта возможных локаций")
st.pyplot(fig, use_container_width=True)

st.subheader("Топ-5 координат")
if not top_points:
st.info("Подходящих точек не найдено. Попробуй увеличить порог или шаги сетки.")
else:
for k, p in enumerate(top_points, start=1):
st.write(f"{k}) Широта: {p['lat']:.2f}°, Долгота: {p['lon']:.2f}°, Разница: {p['diff']:.2%}")

st.caption("Важно: фото используется только для отображения. Авто-определение параметров по фото не выполняется.")
except Exception as e:
st.error(f"Ошибка: {e}")
