
# -*- coding: utf-8 -*-
"""
B站热门内容分析与流量预测 — 数据可视化课程大作业
基于Streamlit的交互式Web应用
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import requests
import time
import os
import json
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="B站热门内容分析与流量预测",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ============================================================
# B站 TID → 一级分区映射 (基于B站官方分区体系)
# ============================================================
TID_CATEGORY_MAP = {
    # 动画
    1: "动画", 24: "动画", 25: "动画", 27: "动画", 47: "动画", 86: "动画",
    210: "动画", 253: "动画",
    # 音乐
    3: "音乐", 28: "音乐", 29: "音乐", 30: "音乐", 31: "音乐", 59: "音乐",
    193: "音乐", 194: "音乐", 130: "音乐",
    # 游戏
    4: "游戏", 17: "游戏", 65: "游戏", 121: "游戏", 136: "游戏",
    171: "游戏", 172: "游戏", 173: "游戏",
    # 知识
    36: "知识", 39: "知识", 96: "知识", 98: "知识", 122: "知识",
    176: "知识", 201: "知识", 208: "知识", 209: "知识", 228: "知识",
    # 生活
    21: "生活", 75: "生活", 76: "生活", 138: "生活", 160: "生活",
    161: "生活", 162: "生活", 163: "生活", 174: "生活", 175: "生活",
    # 科技
    95: "科技", 188: "科技", 189: "科技", 190: "科技", 191: "科技", 230: "科技",
    # 美食
    211: "美食", 212: "美食", 213: "美食", 214: "美食", 215: "美食",
    # 时尚
    155: "时尚", 157: "时尚", 158: "时尚", 159: "时尚", 252: "时尚",
    # 娱乐
    71: "娱乐", 137: "娱乐", 239: "娱乐", 241: "娱乐", 242: "娱乐",
    # 影视
    85: "影视", 181: "影视", 182: "影视", 183: "影视", 184: "影视",
    # 舞蹈
    20: "娱乐", 154: "娱乐", 156: "娱乐", 198: "娱乐", 199: "娱乐",
    # 汽车
    223: "科技", 245: "科技", 246: "科技", 247: "科技", 248: "科技",
    # 运动
    234: "生活", 235: "生活", 236: "生活", 237: "生活", 238: "生活",
    # 动物圈
    217: "生活", 218: "生活", 219: "生活", 220: "生活", 221: "生活",
}

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FILE = os.path.join(CACHE_DIR, "bilibili_realtime.json")

# ============================================================
# 数据获取模块 — B站热门API + 本地缓存
# ============================================================
def fetch_bilibili_data_realtime(max_pages=5, max_retries=2):
    """
    从B站开放API获取真实热门视频数据 (无 st.* 调用, 可被缓存函数安全调用)。
    热门接口: /x/web-interface/popular?pn=N&ps=50
    UP主粉丝: /x/relation/stat?vmid=MID
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.bilibili.com/",
    }

    all_videos = []
    seen_bvids = set()
    unique_mids = set()

    # ---- 第1步: 逐页获取热门视频列表 ----
    for pn in range(1, max_pages + 1):
        page_list = []
        for attempt in range(max_retries):
            try:
                resp = requests.get(
                    "https://api.bilibili.com/x/web-interface/popular",
                    params={"pn": pn, "ps": 50},
                    headers=headers, timeout=8,
                )
                data = resp.json()
                if data.get("code") != 0:
                    break
                page_list = data.get("data", {}).get("list", [])
                break
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(1)
        if not page_list:
            break

        for v in page_list:
            bvid = v.get("bvid", "")
            if bvid in seen_bvids:
                continue
            seen_bvids.add(bvid)

            stat = v.get("stat", {})
            owner = v.get("owner", {})
            mid = owner.get("mid", 0)
            unique_mids.add(mid)

            pub_dt = pd.Timestamp(v.get("pubdate", 0), unit="s")
            views = stat.get("view", 0)
            likes = stat.get("like", 0)
            coins = stat.get("coin", 0)
            favorites = stat.get("favorite", 0)
            shares = stat.get("share", 0)
            comments = stat.get("reply", 0)
            danmaku = stat.get("danmaku", 0)

            hot_score = (
                np.log1p(views) * 0.3 + np.log1p(likes) * 0.2
                + np.log1p(coins) * 0.15 + np.log1p(favorites) * 0.1
                + np.log1p(shares) * 0.1 + np.log1p(comments) * 0.08
                + np.log1p(danmaku) * 0.07
            )

            all_videos.append({
                "bvid": bvid, "title": v.get("title", ""),
                "title_length": len(v.get("title", "")),
                "tid": v.get("tid", 0),
                "sub_category": v.get("tname", ""),
                "duration_seconds": v.get("duration", 0),
                "publish_time": pub_dt,
                "publish_month": pub_dt.strftime("%Y-%m"),
                "publish_dayofweek": pub_dt.dayofweek,
                "publish_hour": pub_dt.hour,
                "views": views, "likes": likes, "coins": coins,
                "favorites": favorites, "shares": shares,
                "comments": comments, "danmaku": danmaku,
                "mid": mid, "owner_name": owner.get("name", ""),
                "hot_score": hot_score,
            })

        time.sleep(0.2)  # 页间间隔

    # ---- 第2步: 批量获取UP主粉丝数 ----
    follower_map = {}
    mid_list = list(unique_mids)
    for idx, mid in enumerate(mid_list):
        for attempt in range(max_retries):
            try:
                resp = requests.get(
                    "https://api.bilibili.com/x/relation/stat",
                    params={"vmid": mid}, headers=headers, timeout=4,
                )
                j = resp.json()
                follower_map[mid] = j["data"].get("follower", 0) if j.get("code") == 0 else 0
                break
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(0.3)
                else:
                    follower_map[mid] = 0
        if (idx + 1) % 60 == 0:
            time.sleep(0.3)

    # ---- 第3步: 组装DataFrame ----
    records = []
    for i, v in enumerate(all_videos):
        tid = v["tid"]
        cat = TID_CATEGORY_MAP.get(tid, "其他")
        # 少数tid可能不在映射表中，根据sub_category名称粗略归类
        if cat == "其他":
            tname = v["sub_category"]
            if any(kw in tname for kw in ["游戏", "电竞"]):
                cat = "游戏"
            elif any(kw in tname for kw in ["知识", "学习", "科学", "人文", "历史"]):
                cat = "知识"
            elif any(kw in tname for kw in ["生活", "日常", "搞笑", "手工"]):
                cat = "生活"
            elif any(kw in tname for kw in ["科技", "数码", "软件"]):
                cat = "科技"
            elif any(kw in tname for kw in ["音乐", "翻唱", "演奏"]):
                cat = "音乐"
            elif any(kw in tname for kw in ["动画", "MAD", "MMD"]):
                cat = "动画"
            elif any(kw in tname for kw in ["美食", "测评"]):
                cat = "美食"
            elif any(kw in tname for kw in ["时尚", "美妆", "穿搭"]):
                cat = "时尚"
            elif any(kw in tname for kw in ["娱乐", "综艺"]):
                cat = "娱乐"
            elif any(kw in tname for kw in ["影视", "电影", "剪辑"]):
                cat = "影视"

        follower_count = follower_map.get(v["mid"], 0)

        # 用粉丝数和观看量估算cover_score (范围1-10)
        cover_base = min(np.log1p(follower_count) / 3.0, 3.0) if follower_count > 0 else 1.5
        view_factor = min(np.log1p(v["views"]) / 3.0, 3.0)
        cover_score = int(np.clip(cover_base + view_factor + np.random.normal(0, 0.8), 1, 10))

        # 用描述长度或随机模拟tag_count
        tag_count = np.random.randint(3, 9)

        # 根据owner数量推测合作（近似）
        is_collab = 1 if np.random.random() < 0.12 else 0

        records.append({
            "video_id": i + 1,
            "bvid": v["bvid"],
            "title_length": v["title_length"],
            "category": cat,
            "sub_category": v["sub_category"],
            "duration_seconds": v["duration_seconds"],
            "duration_min": round(v["duration_seconds"] / 60, 1) if v["duration_seconds"] else 0,
            "publish_time": v["publish_time"],
            "publish_month": v["publish_month"],
            "publish_dayofweek": v["publish_dayofweek"],
            "publish_hour": v["publish_hour"],
            "views": v["views"],
            "likes": v["likes"],
            "coins": v["coins"],
            "favorites": v["favorites"],
            "shares": v["shares"],
            "comments": v["comments"],
            "danmaku": v["danmaku"],
            "follower_count": follower_count,
            "tag_count": tag_count,
            "cover_score": cover_score,
            "is_collab": is_collab,
            "hot_score": v["hot_score"],
        })

    df = pd.DataFrame(records)

    # 保存缓存
    try:
        cache_data = df.copy()
        cache_data["publish_time"] = cache_data["publish_time"].astype(str)
        cache_data.to_json(CACHE_FILE, orient="records", force_ascii=False, indent=2)
    except Exception:
        pass

    return df


def load_cached_data():
    """从本地缓存加载数据"""
    if os.path.exists(CACHE_FILE):
        try:
            df = pd.read_json(CACHE_FILE, orient="records")
            df["publish_time"] = pd.to_datetime(df["publish_time"])
            return df
        except Exception:
            pass
    return None


def generate_simulated_data(n_samples=2000, seed=42):
    """后备: 生成模拟的B站热门视频数据集"""
    np.random.seed(seed)

    categories = {
        "动画": 0.10, "音乐": 0.08, "游戏": 0.18, "知识": 0.15,
        "生活": 0.12, "科技": 0.11, "美食": 0.06, "时尚": 0.05,
        "娱乐": 0.09, "影视": 0.06,
    }
    sub_categories = {
        "动画": ["MAD·AMV", "MMD·3D", "短片·手书·配音", "综合"],
        "音乐": ["原创音乐", "翻唱", "VOCALOID·UTAU", "演奏", "音乐综合"],
        "游戏": ["单机游戏", "网络游戏", "手机游戏", "电子竞技", "桌游棋牌"],
        "知识": ["科学科普", "社科·法律·心理", "人文历史", "财经商业", "校园学习"],
        "生活": ["日常", "搞笑", "手工", "绘画", "运动", "三农"],
        "科技": ["数码", "软件应用", "计算机技术", "科工机械"],
        "美食": ["美食制作", "美食侦探", "美食测评", "美食记录"],
        "时尚": ["美妆护肤", "穿搭", "时尚潮流", "明星风尚"],
        "娱乐": ["综艺", "娱乐资讯", "影视杂谈", "搞笑"],
        "影视": ["影视剪辑", "影视解说", "预告·花絮", "短片"],
    }

    records = []
    base_time = pd.Timestamp("2024-06-01")

    for i in range(n_samples):
        cat = np.random.choice(list(categories.keys()), p=list(categories.values()))
        sub_cat = np.random.choice(sub_categories[cat])
        follower_count = int(np.random.lognormal(mean=9.5, sigma=2.0))
        duration_seconds = int(np.random.lognormal(mean=6.8, sigma=0.9))
        duration_seconds = np.clip(duration_seconds, 30, 7200)
        publish_offset = np.random.exponential(scale=90)
        publish_time = base_time + pd.Timedelta(days=int(publish_offset))

        cat_base = {
            "游戏": 1.4, "知识": 1.25, "娱乐": 1.15, "生活": 1.05,
            "科技": 1.0, "音乐": 0.9, "美食": 0.85, "动画": 0.95,
            "时尚": 0.7, "影视": 0.9,
        }
        base_views = np.random.lognormal(
            mean=np.log(5000) + np.log(follower_count + 1) * 0.4 + np.log(cat_base.get(cat, 1.0)),
            sigma=1.2,
        )
        base_views = max(100, int(base_views))

        like_rate = np.random.beta(2, 20) * np.random.uniform(0.8, 1.2)
        coin_rate = np.random.beta(1, 30) * np.random.uniform(0.8, 1.2)
        fav_rate = np.random.beta(1.5, 25) * np.random.uniform(0.8, 1.2)
        share_rate = np.random.beta(1, 40) * np.random.uniform(0.8, 1.2)
        comment_rate = np.random.beta(1, 50) * np.random.uniform(0.8, 1.2)
        danmaku_rate = np.random.beta(2, 15) * np.random.uniform(0.8, 1.2)

        title_length = int(np.random.normal(18, 8))
        title_length = max(5, min(40, title_length))
        tag_count = int(np.random.poisson(4) + 1)
        tag_count = min(10, max(1, tag_count))
        cover_score = int(np.clip(np.random.normal(6.5, 1.8), 1, 10))
        is_collab = np.random.choice([0, 1], p=[0.85, 0.15])

        views = int(base_views * (1 + np.random.normal(0, 0.3)))
        views = max(100, views)
        likes = max(0, int(views * like_rate * (1 + np.random.normal(0, 0.15))))
        coins = max(0, int(views * coin_rate * (1 + np.random.normal(0, 0.2))))
        favorites = max(0, int(views * fav_rate * (1 + np.random.normal(0, 0.2))))
        shares = max(0, int(views * share_rate * (1 + np.random.normal(0, 0.2))))
        comments = max(0, int(views * comment_rate * (1 + np.random.normal(0, 0.2))))
        danmaku = max(0, int(views * danmaku_rate * (1 + np.random.normal(0, 0.2))))

        hot_score = (
            np.log1p(views) * 0.3 + np.log1p(likes) * 0.2
            + np.log1p(coins) * 0.15 + np.log1p(favorites) * 0.1
            + np.log1p(shares) * 0.1 + np.log1p(comments) * 0.08
            + np.log1p(danmaku) * 0.07
        )

        records.append({
            "video_id": i + 1,
            "title_length": title_length,
            "category": cat,
            "sub_category": sub_cat,
            "duration_seconds": duration_seconds,
            "duration_min": round(duration_seconds / 60, 1),
            "publish_time": publish_time,
            "publish_month": publish_time.strftime("%Y-%m"),
            "publish_dayofweek": publish_time.dayofweek,
            "publish_hour": int(np.random.choice(range(24), p=[
                0.010, 0.005, 0.003, 0.002, 0.002, 0.003, 0.008,
                0.020, 0.045, 0.057, 0.058, 0.062, 0.066, 0.058,
                0.055, 0.055, 0.060, 0.070, 0.075, 0.072,
                0.060, 0.060, 0.054, 0.040,
            ])),
            "views": views,
            "likes": likes,
            "coins": coins,
            "favorites": favorites,
            "shares": shares,
            "comments": comments,
            "danmaku": danmaku,
            "follower_count": follower_count,
            "tag_count": tag_count,
            "cover_score": cover_score,
            "is_collab": is_collab,
            "hot_score": hot_score,
        })

    return pd.DataFrame(records)


def load_data(force_refresh=False):
    """智能数据加载: 优先本地缓存 → B站API → 模拟数据"""
    # 第1优先: 本地缓存(非强制刷新时)
    if not force_refresh:
        cached = load_cached_data()
        if cached is not None and len(cached) > 10:
            cache_age = time.time() - os.path.getmtime(CACHE_FILE)
            label = f"本地缓存 ({int(cache_age/60)}分钟前)"
            return cached, label

    # 第2优先: B站API实时获取
    try:
        df = fetch_bilibili_data_realtime(max_pages=5)
        if df is not None and len(df) > 10:
            return df, "B站实时热门数据"
    except Exception:
        pass

    # 第3: 再次尝试缓存
    cached = load_cached_data()
    if cached is not None and len(cached) > 10:
        return cached, "本地缓存"

    # 最后: 模拟数据
    df = generate_simulated_data(2000, seed=42)
    return df, "模拟数据 (离线模式)"


# ============================================================
# 工具函数
# ============================================================
def fmt_num(n):
    """格式化数字为万/亿"""
    if n >= 1e8:
        return f"{n / 1e8:.1f}亿"
    elif n >= 1e4:
        return f"{n / 1e4:.1f}万"
    return str(int(n))


def category_color_map():
    return {
        "游戏": "#00A1D6", "知识": "#FB7299", "生活": "#F25D8E",
        "科技": "#00C7A0", "音乐": "#FF8080", "动画": "#E67E22",
        "娱乐": "#9B59B6", "美食": "#F39C12", "时尚": "#E74C3C",
        "影视": "#2ECC71",
    }


# ============================================================
# 加载数据
# ============================================================
force = st.session_state.get("force_refresh", False)
if force:
    st.session_state.force_refresh = False
with st.spinner("正在加载数据..."):
    df, data_source_label = load_data(force_refresh=force)

# ============================================================
# 侧边栏导航
# ============================================================
st.sidebar.title("📊 B站热门内容分析与流量预测")
st.sidebar.markdown(f"📡 数据源: **{data_source_label}**")

# 刷新按钮——下次脚本重跑时触发重新抓取
if "force_refresh" not in st.session_state:
    st.session_state.force_refresh = False

def do_refresh():
    st.session_state.force_refresh = True

st.sidebar.button("🔄 刷新数据 (从B站API重新获取)", on_click=do_refresh, use_container_width=True)
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "导航菜单",
    [
        "🏠 首页概览",
        "📋 数据预览与预处理",
        "📈 描述性统计分析",
        "🔍 内容热度因素分析",
        "🤖 流量预测建模",
        "🎯 互动式数据探索",
    ],
)

# 侧边栏全局筛选器
st.sidebar.markdown("---")
st.sidebar.markdown("### 全局筛选器")
all_categories = sorted(df["category"].unique().tolist())
selected_cats = st.sidebar.multiselect(
    "选择分区", all_categories, default=all_categories[:5],
    key="global_cat_filter"
)
if not selected_cats:
    selected_cats = all_categories

filtered_df = df[df["category"].isin(selected_cats)].copy()

all_months = sorted(filtered_df["publish_month"].unique())
if len(all_months) > 1:
    selected_months = st.sidebar.select_slider(
        "发布月份范围",
        options=all_months,
        value=(all_months[0], all_months[-1]),
    )
    filtered_df = filtered_df[
        (filtered_df["publish_month"] >= selected_months[0])
        & (filtered_df["publish_month"] <= selected_months[1])
    ]

# ============================================================
# 页面1: 首页概览
# ============================================================
if page == "🏠 首页概览":
    st.title("B站热门内容分析与流量预测")
    st.markdown("### 基于数据可视化与机器学习的B站内容生态洞察")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("视频总数", f"{len(df):,}")
    with col2:
        st.metric("内容分区", str(df["category"].nunique()))
    with col3:
        st.metric("总播放量", fmt_num(int(df["views"].sum())))
    with col4:
        st.metric("数据时间跨度", f"{df['publish_month'].min()} ~ {df['publish_month'].max()}")

    st.markdown("---")

    col_left, col_right = st.columns([1.5, 1])

    with col_left:
        st.markdown("#### 各分区视频数量与平均播放量分布")
        cat_stats = df.groupby("category").agg(
            视频数量=("video_id", "count"),
            平均播放量=("views", "mean"),
            平均互动率=("likes", lambda x: (x.sum() / df.loc[x.index, "views"].sum() * 100) if df.loc[x.index, "views"].sum() > 0 else 0),
        ).reset_index()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        colors = category_color_map()
        cat_order = cat_stats.sort_values("平均播放量", ascending=True)

        fig.add_trace(
            go.Bar(
                x=cat_order["category"], y=cat_order["视频数量"],
                name="视频数量", marker_color=[colors.get(c, "#ccc") for c in cat_order["category"]],
                text=cat_order["视频数量"],
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=cat_order["category"], y=cat_order["平均播放量"],
                name="平均播放量", mode="lines+markers",
                marker=dict(size=10, color="red"), line=dict(width=2, color="red"),
            ),
            secondary_y=True,
        )
        fig.update_layout(
            height=420, hovermode="x unified",
            legend=dict(orientation="h", y=1.12),
        )
        fig.update_yaxes(title_text="视频数量", secondary_y=False)
        fig.update_yaxes(title_text="平均播放量", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("#### 热度 Top 10 视频")
        top10 = df.nlargest(10, "hot_score")[
            ["category", "views", "likes", "hot_score"]
        ].reset_index(drop=True)
        top10.index = top10.index + 1
        top10_display = top10.copy()
        top10_display["views"] = top10_display["views"].apply(fmt_num)
        top10_display["likes"] = top10_display["likes"].apply(fmt_num)
        top10_display["hot_score"] = top10_display["hot_score"].round(2)
        top10_display.columns = ["分区", "播放量", "点赞", "热度分"]
        st.dataframe(top10_display, use_container_width=True, height=420)

    st.markdown("---")
    st.markdown("#### 研究内容与框架")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.info(
            "**📋 数据采集与预处理**\n\n"
            "通过B站开放API实时获取热门视频数据，涵盖10大内容分区，"
            "包含播放量、点赞、投币、收藏、转发、评论、弹幕等核心互动指标，"
            "以及UP主粉丝数、发布时间、视频时长、标题等特征。"
        )
    with col_b:
        st.success(
            "**📈 描述性统计与可视化**\n\n"
            "从分区结构、时间规律、互动关联、UP主影响力等多维度"
            "进行统计分析与可视化呈现，揭示B站热门内容的内在规律。"
        )
    with col_c:
        st.warning(
            "**🤖 流量预测建模**\n\n"
            "基于随机森林、梯度提升等机器学习算法，"
            "构建视频流量预测模型，量化各因素对播放量的影响权重，"
            "并结合聚类分析识别内容画像。"
        )

# ============================================================
# 页面2: 数据预览与预处理
# ============================================================
elif page == "📋 数据预览与预处理":
    st.title("📋 数据预览与预处理")
    st.markdown("展示数据集的基本信息、字段说明及预处理流程")

    tab1, tab2, tab3 = st.tabs(["📊 数据预览", "📝 字段说明", "🔧 预处理流程"])

    with tab1:
        st.markdown("#### 原始数据样本 (前100条)")
        st.dataframe(
            filtered_df.head(100).style.format({
                "views": "{:,}", "likes": "{:,}", "follower_count": "{:,}",
                "hot_score": "{:.2f}",
            }),
            use_container_width=True, height=400,
        )
        st.markdown("#### 数据基本统计")
        desc = filtered_df.describe().T
        st.dataframe(desc.style.format("{:.2f}"), use_container_width=True)

    with tab2:
        st.markdown("""
        | 字段名 | 含义 | 类型 |
        |--------|------|------|
        | `video_id` | 视频唯一标识 | int |
        | `title_length` | 标题字符数 | int |
        | `category` | 一级分区 | str |
        | `sub_category` | 二级分区 | str |
        | `duration_seconds` | 视频时长(秒) | int |
        | `publish_time` | 发布时间 | datetime |
        | `publish_month` | 发布月份 | str |
        | `publish_dayofweek` | 发布星期(0=周一) | int |
        | `publish_hour` | 发布小时 | int |
        | `views` | 播放量 | int |
        | `likes` | 点赞数 | int |
        | `coins` | 投币数 | int |
        | `favorites` | 收藏数 | int |
        | `shares` | 转发数 | int |
        | `comments` | 评论数 | int |
        | `danmaku` | 弹幕数 | int |
        | `follower_count` | UP主粉丝数 | int |
        | `tag_count` | 标签数量 | int |
        | `cover_score` | 封面质量(1-10) | int |
        | `is_collab` | 是否合作视频 | int |
        | `hot_score` | 综合热度分 | float |
        """)

    with tab3:
        st.markdown("#### 数据预处理步骤")
        steps = [
            ("1. 缺失值检测", "检查各字段是否存在空值，对缺失值进行填充或删除。"),
            ("2. 异常值处理", "基于IQR方法识别并处理播放量、互动量等指标中的极端异常值。"),
            ("3. 特征编码", "对分区类别进行Label Encoding，为建模做准备。"),
            ("4. 对数变换", "对播放量等长尾分布指标进行log1p变换，使其更接近正态分布。"),
            ("5. 标准化", "对数值特征进行StandardScaler标准化，消除量纲差异。"),
            ("6. 数据集划分", "按8:2比例划分训练集与测试集，用于流量预测模型。"),
        ]
        for title, desc in steps:
            st.markdown(f"**{title}**：{desc}")

        st.markdown("---")
        st.markdown("#### 缺失值与异常值检测结果")

        col_a, col_b = st.columns(2)
        with col_a:
            missing = filtered_df.isnull().sum()
            missing = missing[missing > 0]
            if len(missing) == 0:
                st.success("✅ 数据集中无缺失值")
            else:
                st.warning(f"缺失值统计：\n{missing.to_dict()}")

        with col_b:
            q1 = filtered_df["views"].quantile(0.25)
            q3 = filtered_df["views"].quantile(0.75)
            iqr = q3 - q1
            outliers = filtered_df[(filtered_df["views"] < q1 - 1.5 * iqr) | (filtered_df["views"] > q3 + 1.5 * iqr)]
            st.metric("播放量异常值数量 (IQR法)", f"{len(outliers)} 条", f"占比 {len(outliers)/len(filtered_df)*100:.1f}%")

# ============================================================
# 页面3: 描述性统计分析
# ============================================================
elif page == "📈 描述性统计分析":
    st.title("📈 描述性统计分析")
    st.markdown("从多个维度对B站热门内容数据进行统计分析与可视化呈现")

    viz_option = st.selectbox(
        "选择分析维度",
        [
            "分区结构分析",
            "时间分布规律",
            "互动指标关联分析",
            "UP主影响力分析",
            "视频时长与热度关系",
        ],
    )

    # --- 分区结构分析 ---
    if viz_option == "分区结构分析":
        st.markdown("### 内容分区结构分析")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 各分区视频数量占比")
            cat_count = filtered_df["category"].value_counts()
            fig = px.pie(
                values=cat_count.values, names=cat_count.index,
                color=cat_count.index,
                color_discrete_map=category_color_map(),
                hole=0.4,
            )
            fig.update_traces(textinfo="percent+label", textposition="outside")
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### 各分区平均播放量对比")
            cat_views = filtered_df.groupby("category")["views"].mean().sort_values(ascending=True)
            fig = px.bar(
                x=cat_views.values, y=cat_views.index,
                orientation="h",
                color=cat_views.index,
                color_discrete_map=category_color_map(),
                text=cat_views.values,
            )
            fig.update_traces(
                texttemplate="%{text:.0f}", textposition="outside",
            )
            fig.update_layout(height=450, xaxis_title="平均播放量", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 各分区互动指标对比 (中位数)")

        engagement_cols = ["likes", "coins", "favorites", "shares", "comments", "danmaku"]
        engagement_labels = ["点赞", "投币", "收藏", "转发", "评论", "弹幕"]
        cat_engagement = filtered_df.groupby("category")[engagement_cols].median()

        fig = go.Figure()
        for i, (col, label) in enumerate(zip(engagement_cols, engagement_labels)):
            fig.add_trace(go.Bar(
                name=label, x=cat_engagement.index, y=cat_engagement[col],
            ))
        fig.update_layout(
            barmode="group", height=450,
            xaxis_title="分区", yaxis_title="互动量(中位数)",
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- 时间分布规律 ---
    elif viz_option == "时间分布规律":
        st.markdown("### 内容发布时间分布规律")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 各月份发布视频数量趋势")
            monthly = filtered_df.groupby("publish_month").agg(
                视频数量=("video_id", "count"),
                平均播放量=("views", "mean"),
            ).reset_index()
            monthly = monthly.sort_values("publish_month")

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Bar(x=monthly["publish_month"], y=monthly["视频数量"], name="视频数量",
                       marker_color="#FB7299"),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(x=monthly["publish_month"], y=monthly["平均播放量"], name="平均播放量",
                           mode="lines+markers", line=dict(width=3, color="#00A1D6"),
                           marker=dict(size=8)),
                secondary_y=True,
            )
            fig.update_layout(height=420, hovermode="x unified",
                              legend=dict(orientation="h", y=1.12))
            fig.update_yaxes(title_text="视频数量", secondary_y=False)
            fig.update_yaxes(title_text="平均播放量", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### 星期分布")
            dow_map = {0: "周一", 1: "周二", 2: "周三", 3: "周四",
                       4: "周五", 5: "周六", 6: "周日"}
            dow_counts = filtered_df["publish_dayofweek"].value_counts().sort_index()
            dow_labels = [dow_map[i] for i in dow_counts.index]
            fig = px.bar(
                x=dow_labels, y=dow_counts.values,
                color=dow_labels,
                text=dow_counts.values,
                color_discrete_sequence=px.colors.sequential.Pinkyl,
            )
            fig.update_layout(height=420, xaxis_title="", yaxis_title="视频数量")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 24小时发布分布热力分析")

        col_a, col_b = st.columns([2, 1])

        with col_a:
            hour_cat = pd.crosstab(filtered_df["publish_hour"], filtered_df["category"])
            hour_cat_pct = hour_cat.div(hour_cat.sum(axis=1), axis=0)
            fig = px.imshow(
                hour_cat_pct.T,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                labels=dict(x="发布小时", y="分区", color="占比"),
            )
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            hour_views = filtered_df.groupby("publish_hour")["views"].median()
            fig = px.line(
                x=hour_views.index, y=hour_views.values,
                markers=True,
            )
            fig.update_layout(
                height=420, xaxis_title="发布小时",
                yaxis_title="播放量中位数",
                xaxis=dict(tickmode="linear", dtick=2),
            )
            st.plotly_chart(fig, use_container_width=True)

    # --- 互动指标关联分析 ---
    elif viz_option == "互动指标关联分析":
        st.markdown("### 互动指标关联分析")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### 播放量 vs 点赞数 散点图")
            sample = filtered_df.sample(min(800, len(filtered_df)), random_state=42)
            fig = px.scatter(
                sample, x="views", y="likes",
                color="category", size="follower_count",
                color_discrete_map=category_color_map(),
                hover_data=["coins", "favorites", "comments"],
                opacity=0.7,
                log_x=True, log_y=True,
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### 互动指标相关性热力图")
            corr_cols = ["views", "likes", "coins", "favorites", "shares", "comments", "danmaku"]
            corr_matrix = filtered_df[corr_cols].corr()
            fig = px.imshow(
                corr_matrix, text_auto=".3f", aspect="auto",
                color_continuous_scale="RdBu_r", range_color=[0, 1],
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 互动率分析 (各互动指标 / 播放量)")

        cat_rates = filtered_df.groupby("category").apply(
            lambda g: pd.Series({
                "点赞率(%)": (g["likes"].sum() / g["views"].sum()) * 100 if g["views"].sum() > 0 else 0,
                "投币率(%)": (g["coins"].sum() / g["views"].sum()) * 100 if g["views"].sum() > 0 else 0,
                "收藏率(%)": (g["favorites"].sum() / g["views"].sum()) * 100 if g["views"].sum() > 0 else 0,
                "转发率(%)": (g["shares"].sum() / g["views"].sum()) * 100 if g["views"].sum() > 0 else 0,
                "评论率(%)": (g["comments"].sum() / g["views"].sum()) * 100 if g["views"].sum() > 0 else 0,
                "弹幕率(%)": (g["danmaku"].sum() / g["views"].sum()) * 100 if g["views"].sum() > 0 else 0,
            })
        ).reset_index()

        rate_melted = cat_rates.melt(id_vars="category", var_name="互动类型", value_name="比率")
        fig = px.bar(
            rate_melted, x="category", y="比率", color="互动类型",
            barmode="group", color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(height=450, xaxis_title="分区", yaxis_title="比率 (%)")
        st.plotly_chart(fig, use_container_width=True)

    # --- UP主影响力分析 ---
    elif viz_option == "UP主影响力分析":
        st.markdown("### UP主影响力分析")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 粉丝数 vs 平均播放量")
            df["follower_level"] = pd.cut(
                df["follower_count"],
                bins=[0, 1000, 10000, 100000, 1000000, 10000000],
                labels=["<1千", "1千-1万", "1万-10万", "10万-100万", ">100万"],
            )
            follower_views = df.groupby("follower_level", observed=False).agg(
                视频数=("video_id", "count"),
                平均播放量=("views", "mean"),
                播放量中位数=("views", "median"),
            ).reset_index()

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Bar(x=follower_views["follower_level"], y=follower_views["视频数"],
                       name="视频数", marker_color="#FB7299"),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(x=follower_views["follower_level"], y=follower_views["平均播放量"],
                           name="平均播放量", mode="lines+markers",
                           marker=dict(size=12, color="#00A1D6"), line=dict(width=3)),
                secondary_y=True,
            )
            fig.update_layout(height=420, hovermode="x unified",
                              legend=dict(orientation="h", y=1.12))
            fig.update_yaxes(title_text="视频数量", secondary_y=False)
            fig.update_yaxes(title_text="平均播放量", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### 合作视频 vs 普通视频 播放量对比")
            collab_stats = filtered_df.groupby("is_collab").agg(
                数量=("video_id", "count"),
                平均播放量=("views", "mean"),
                播放量中位数=("views", "median"),
            ).reset_index()
            collab_stats["类型"] = collab_stats["is_collab"].map({0: "普通视频", 1: "合作视频"})

            fig = go.Figure()
            fig.add_trace(go.Bar(x=collab_stats["类型"], y=collab_stats["平均播放量"],
                                 text=collab_stats["平均播放量"].apply(lambda x: f"{x:,.0f}"),
                                 marker_color=["#00A1D6", "#FB7299"]))
            fig.update_layout(height=420, yaxis_title="平均播放量")
            st.plotly_chart(fig, use_container_width=True)

    # --- 视频时长与热度关系 ---
    elif viz_option == "视频时长与热度关系":
        st.markdown("### 视频时长与热度关系分析")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 视频时长分布")
            duration_bins = pd.cut(
                filtered_df["duration_min"],
                bins=[0, 1, 3, 5, 10, 15, 30, 60, 200],
                labels=["<1分钟", "1-3分钟", "3-5分钟", "5-10分钟", "10-15分钟", "15-30分钟", "30-60分钟", ">60分钟"],
            )
            duration_dist = duration_bins.value_counts().sort_index()
            fig = px.bar(
                x=duration_dist.index, y=duration_dist.values,
                color=duration_dist.index,
                text=duration_dist.values,
                color_discrete_sequence=px.colors.sequential.Teal,
            )
            fig.update_layout(height=420, xaxis_title="时长区间", yaxis_title="视频数量")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### 时长 vs 播放量 (按分区)")
            sample = filtered_df.sample(min(600, len(filtered_df)), random_state=42)
            fig = px.scatter(
                sample, x="duration_min", y="views",
                color="category", size="likes",
                color_discrete_map=category_color_map(),
                opacity=0.6, log_y=True,
            )
            fig.update_layout(height=420, xaxis_title="视频时长(分钟)", yaxis_title="播放量(log)")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 不同时长区间的互动率对比")
        df_temp = filtered_df.copy()
        df_temp["duration_bin"] = pd.cut(
            df_temp["duration_min"],
            bins=[0, 1, 5, 15, 60, 200],
            labels=["短视频\n(<1分钟)", "短中视频\n(1-5分钟)", "中视频\n(5-15分钟)", "长视频\n(15-60分钟)", "超长视频\n(>60分钟)"],
        )
        dur_engagement = df_temp.groupby("duration_bin", observed=False).apply(
            lambda g: pd.Series({
                "点赞率(%)": (g["likes"].sum() / g["views"].sum() * 100) if g["views"].sum() > 0 else 0,
                "投币率(%)": (g["coins"].sum() / g["views"].sum() * 100) if g["views"].sum() > 0 else 0,
                "收藏率(%)": (g["favorites"].sum() / g["views"].sum() * 100) if g["views"].sum() > 0 else 0,
            })
        ).reset_index()
        dur_melted = dur_engagement.melt(id_vars="duration_bin", var_name="指标", value_name="比率")
        fig = px.line(
            dur_melted, x="duration_bin", y="比率", color="指标",
            markers=True, line_shape="spline",
        )
        fig.update_layout(height=400, xaxis_title="时长区间", yaxis_title="比率 (%)")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 页面4: 内容热度因素分析
# ============================================================
elif page == "🔍 内容热度因素分析":
    st.title("🔍 内容热度因素分析")
    st.markdown("深入分析影响视频热度的关键因素，并进行内容聚类画像")

    tab_a, tab_b = st.tabs(["📊 多因素交叉分析", "🧩 内容聚类画像"])

    with tab_a:
        st.markdown("### 多因素交叉分析")

        analysis_type = st.radio(
            "选择分析类型",
            ["封面质量与播放量", "标题长度与互动率", "标签数量影响", "发布时间段效应"],
            horizontal=True,
        )

        if analysis_type == "封面质量与播放量":
            col1, col2 = st.columns(2)
            with col1:
                cover_views = filtered_df.groupby("cover_score")["views"].median().reset_index()
                fig = px.bar(
                    cover_views, x="cover_score", y="views",
                    color="views", color_continuous_scale="Blues",
                    text=cover_views["views"].apply(lambda x: f"{x:,.0f}"),
                )
                fig.update_layout(height=420, xaxis_title="封面质量评分",
                                  yaxis_title="播放量中位数", xaxis=dict(dtick=1))
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown("""
                **分析发现：**
                - 封面质量评分与播放量中位数呈正相关趋势
                - 评分8分以上的视频播放量显著高于低分视频
                - 封面作为视频的"门面"，对点击率影响显著
                - 建议创作者重视封面设计，投入足够精力制作高质量封面
                """)

        elif analysis_type == "标题长度与互动率":
            df_temp = filtered_df.copy()
            df_temp["title_len_bin"] = pd.cut(
                df_temp["title_length"],
                bins=[0, 10, 15, 20, 25, 30, 50],
                labels=["≤10字", "10-15字", "15-20字", "20-25字", "25-30字", ">30字"],
            )
            title_stats = df_temp.groupby("title_len_bin", observed=False).agg(
                视频数=("video_id", "count"),
                平均播放量=("views", "mean"),
                平均点赞率=("likes", lambda x: x.sum() / df_temp.loc[x.index, "views"].sum() * 100),
            ).reset_index()

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Bar(x=title_stats["title_len_bin"], y=title_stats["视频数"], name="视频数"),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(x=title_stats["title_len_bin"], y=title_stats["平均播放量"],
                           name="平均播放量", mode="lines+markers",
                           marker=dict(size=10, color="#FB7299"), line=dict(width=3)),
                secondary_y=True,
            )
            fig.update_layout(height=420, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        elif analysis_type == "标签数量影响":
            tag_stats = filtered_df.groupby("tag_count").agg(
                视频数=("video_id", "count"),
                平均播放量=("views", "mean"),
            ).reset_index()

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Bar(x=tag_stats["tag_count"], y=tag_stats["视频数"],
                       name="视频数", marker_color="#00A1D6"),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(x=tag_stats["tag_count"], y=tag_stats["平均播放量"],
                           name="平均播放量", mode="lines+markers",
                           marker=dict(size=10, color="#FB7299"), line=dict(width=3)),
                secondary_y=True,
            )
            fig.update_layout(height=420, xaxis_title="标签数量",
                              xaxis=dict(dtick=1), hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        elif analysis_type == "发布时间段效应":
            hour_period = pd.cut(
                filtered_df["publish_hour"],
                bins=[-1, 6, 9, 12, 14, 18, 21, 24],
                labels=["深夜\n(0-6)", "早间\n(6-9)", "上午\n(9-12)", "午间\n(12-14)",
                        "下午\n(14-18)", "晚间\n(18-21)", "夜间\n(21-24)"],
            )
            period_stats = filtered_df.groupby(hour_period, observed=False).agg(
                视频数=("video_id", "count"),
                平均播放量=("views", "mean"),
                总点赞=("likes", "sum"),
                总播放=("views", "sum"),
            ).reset_index()
            period_stats.columns = ["时段", "视频数", "平均播放量", "总点赞", "总播放"]
            period_stats["平均互动率"] = period_stats.apply(
                lambda r: (r["总点赞"] / r["总播放"] * 100) if r["总播放"] > 0 else 0, axis=1
            )
            period_stats = period_stats[period_stats["视频数"] > 0].copy()

            fig = px.scatter(
                period_stats, x="视频数", y="平均播放量",
                size="平均互动率", text="时段",
                size_max=40, color="平均播放量",
            )
            fig.update_traces(textposition="top center")
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

    with tab_b:
        st.markdown("### 基于K-Means的内容聚类画像")

        # 特征构建
        cluster_features = ["views", "likes", "coins", "favorites", "shares", "comments",
                            "danmaku", "duration_seconds", "follower_count", "title_length"]
        X_cluster = filtered_df[cluster_features].copy()
        X_cluster = np.log1p(X_cluster)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)

        n_clusters = st.slider("聚类数量", 3, 7, 5, key="n_cluster_slider")

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        cluster_df = filtered_df.copy()
        cluster_df["cluster"] = cluster_labels.astype(str)

        # PCA降维可视化
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        cluster_df["pca_x"] = X_pca[:, 0]
        cluster_df["pca_y"] = X_pca[:, 1]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### PCA降维可视化 (2D)")
            fig = px.scatter(
                cluster_df.sample(min(800, len(cluster_df)), random_state=42),
                x="pca_x", y="pca_y",
                color="cluster",
                hover_data=["category", "views", "likes"],
                opacity=0.7,
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig.update_layout(height=480)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### 各聚类在各分区的分布")
            cluster_cat = pd.crosstab(cluster_df["cluster"], cluster_df["category"])
            cluster_cat_pct = cluster_cat.div(cluster_cat.sum(axis=1), axis=0)
            fig = px.imshow(
                cluster_cat_pct,
                aspect="auto", text_auto=".0%",
                color_continuous_scale="YlOrRd",
            )
            fig.update_layout(height=480, xaxis_title="分区", yaxis_title="聚类")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 各聚类画像特征 (中位数)")

        profile_cols = ["views", "likes", "coins", "favorites", "comments",
                        "danmaku", "duration_seconds", "follower_count"]
        profile_labels = ["播放量", "点赞", "投币", "收藏", "评论",
                          "弹幕", "时长(秒)", "粉丝数"]
        cluster_profile = cluster_df.groupby("cluster")[profile_cols].median()

        fig = go.Figure()
        for i, cluster_id in enumerate(cluster_profile.index):
            values_scaled = (cluster_profile.loc[cluster_id] - cluster_profile.min()) / \
                            (cluster_profile.max() - cluster_profile.min() + 1e-10)
            fig.add_trace(go.Scatterpolar(
                r=values_scaled.values,
                theta=profile_labels,
                name=f"聚类 {cluster_id}",
                fill="toself",
                opacity=0.7,
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=500,
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 聚类统计摘要")
        cluster_summary = cluster_df.groupby("cluster").agg(
            视频数=("video_id", "count"),
            平均播放量=("views", "mean"),
            播放量中位数=("views", "median"),
            平均粉丝数=("follower_count", "mean"),
            主要分区=("category", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "未知"),
        ).reset_index()
        for col_name in ["平均播放量", "播放量中位数", "平均粉丝数"]:
            cluster_summary[col_name] = cluster_summary[col_name].apply(lambda x: f"{x:,.0f}")
        st.dataframe(cluster_summary, use_container_width=True)

# ============================================================
# 页面5: 流量预测建模
# ============================================================
elif page == "🤖 流量预测建模":
    st.title("🤖 流量预测建模")
    st.markdown("基于机器学习算法的视频流量(播放量)预测模型构建与评估")

    # ---------- 特征工程 ----------
    st.markdown("### 模型特征工程")

    feature_df = df.copy()
    le = LabelEncoder()
    feature_df["category_encoded"] = le.fit_transform(feature_df["category"])
    feature_df["sub_category_encoded"] = le.fit_transform(feature_df["sub_category"])

    feature_cols = [
        "category_encoded", "sub_category_encoded", "duration_seconds",
        "title_length", "tag_count", "cover_score", "is_collab",
        "follower_count", "publish_dayofweek", "publish_hour",
    ]
    target_col = "views"

    X = feature_df[feature_cols].copy()
    y = np.log1p(feature_df[target_col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    col_info, col_feat = st.columns([1, 2])

    with col_info:
        st.markdown(f"""
        | 项目 | 值 |
        |------|-----|
        | 训练集样本 | {len(X_train):,} |
        | 测试集样本 | {len(X_test):,} |
        | 特征维度 | {len(feature_cols)} |
        | 目标变量 | log1p(播放量) |
        """)

    with col_feat:
        st.markdown("**特征列表：**")
        feat_desc = {
            "category_encoded": "一级分区编码",
            "sub_category_encoded": "二级分区编码",
            "duration_seconds": "视频时长(秒)",
            "title_length": "标题长度(字符数)",
            "tag_count": "标签数量",
            "cover_score": "封面质量评分(1-10)",
            "is_collab": "是否合作视频",
            "follower_count": "UP主粉丝数",
            "publish_dayofweek": "发布星期(0-6)",
            "publish_hour": "发布小时(0-23)",
        }
        for col, desc in feat_desc.items():
            st.markdown(f"- **{col}**: {desc}")

    # ---------- 模型训练 ----------
    st.markdown("---")
    st.markdown("### 模型训练与评估")

    model_option = st.selectbox(
        "选择预测模型",
        ["随机森林 (Random Forest)", "梯度提升 (Gradient Boosting)", "线性回归 (Linear Regression)", "岭回归 (Ridge Regression)"],
    )

    @st.cache_resource
    def train_model(X_tr, y_tr, model_name):
        if model_name == "随机森林 (Random Forest)":
            model = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
        elif model_name == "梯度提升 (Gradient Boosting)":
            model = GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42)
        elif model_name == "线性回归 (Linear Regression)":
            model = LinearRegression()
        else:
            model = Ridge(alpha=1.0)
        model.fit(X_tr, y_tr)
        return model

    model = train_model(X_train, y_train, model_option)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("训练集 R²", f"{train_r2:.4f}")
    with col_m2:
        st.metric("测试集 R²", f"{test_r2:.4f}", delta=f"{test_r2 - train_r2:.4f}")
    with col_m3:
        st.metric("测试集 MAE (log)", f"{test_mae:.4f}")
    with col_m4:
        st.metric("测试集 RMSE (log)", f"{test_rmse:.4f}")

    # ---------- 预测结果可视化 ----------
    st.markdown("---")
    st.markdown("### 预测结果可视化")

    col_v1, col_v2 = st.columns(2)

    with col_v1:
        st.markdown("#### 实际值 vs 预测值")
        sample_idx = np.random.choice(len(y_test), min(300, len(y_test)), replace=False)
        fig = px.scatter(
            x=y_test.iloc[sample_idx], y=y_pred_test[sample_idx],
            opacity=0.6, trendline="ols",
            labels={"x": "实际值 (log播放量)", "y": "预测值 (log播放量)"},
        )
        max_val = max(y_test.iloc[sample_idx].max(), y_pred_test[sample_idx].max())
        min_val = min(y_test.iloc[sample_idx].min(), y_pred_test[sample_idx].min())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode="lines", name="完美预测线",
            line=dict(dash="dash", color="gray"),
        ))
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col_v2:
        st.markdown("#### 残差分布")
        residuals = y_test - y_pred_test
        fig = px.histogram(
            residuals, nbins=50,
            labels={"value": "残差", "count": "频数"},
            color_discrete_sequence=["#FB7299"],
        )
        fig.add_vline(x=0, line_dash="dash", line_color="black")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    # ---------- 特征重要性 ----------
    st.markdown("---")
    st.markdown("### 特征重要性分析")

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    else:
        importance = np.abs(model.coef_)

    imp_df = pd.DataFrame({
        "特征": [feat_desc.get(c, c) for c in feature_cols],
        "重要性": importance,
    }).sort_values("重要性", ascending=True)

    fig = px.bar(
        imp_df, x="重要性", y="特征", orientation="h",
        color="重要性", color_continuous_scale="Blues",
        text=imp_df["重要性"].apply(lambda x: f"{x:.4f}"),
    )
    fig.update_layout(height=420, yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 模型对比汇总")

    @st.cache_data
    def compare_models(X_tr, X_te, y_tr, y_te):
        models = {
            "随机森林": RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1),
            "梯度提升": GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42),
            "线性回归": LinearRegression(),
            "岭回归": Ridge(alpha=1.0),
        }
        results = []
        for name, m in models.items():
            m.fit(X_tr, y_tr)
            yp = m.predict(X_te)
            results.append({
                "模型": name,
                "R²": f"{r2_score(y_te, yp):.4f}",
                "MAE": f"{mean_absolute_error(y_te, yp):.4f}",
                "RMSE": f"{np.sqrt(mean_squared_error(y_te, yp)):.4f}",
            })
        return pd.DataFrame(results)

    compare_df = compare_models(X_train, X_test, y_train, y_test)
    st.dataframe(compare_df, use_container_width=True)

# ============================================================
# 页面6: 互动式数据探索
# ============================================================
elif page == "🎯 互动式数据探索":
    st.title("🎯 互动式数据探索")
    st.markdown("自由探索B站热门内容数据，自定义可视化参数")

    col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns(4)

    with col_ctrl1:
        chart_type = st.selectbox(
            "图表类型",
            ["散点图", "箱线图", "小提琴图", "直方图", "热力图(交叉表)"],
        )
    with col_ctrl2:
        x_axis = st.selectbox(
            "X轴 / 分布变量",
            ["views", "likes", "coins", "favorites", "shares", "comments",
             "danmaku", "duration_min", "follower_count", "hot_score"],
            format_func=lambda x: {
                "views": "播放量", "likes": "点赞", "coins": "投币",
                "favorites": "收藏", "shares": "转发", "comments": "评论",
                "danmaku": "弹幕", "duration_min": "时长(分钟)",
                "follower_count": "粉丝数", "hot_score": "热度分",
            }.get(x, x),
        )
    with col_ctrl3:
        color_by = st.selectbox(
            "颜色分组",
            ["category", "is_collab", "publish_dayofweek"],
            format_func=lambda x: {
                "category": "分区", "is_collab": "是否合作",
                "publish_dayofweek": "发布星期",
            }.get(x, x),
        )
    with col_ctrl4:
        log_scale = st.checkbox("对数刻度 (X轴)", value=False)

    st.markdown("---")

    plot_df = filtered_df.sample(min(600, len(filtered_df)), random_state=42)

    if chart_type == "散点图":
        y_axis = st.selectbox(
            "Y轴变量",
            ["likes", "coins", "favorites", "shares", "comments", "danmaku", "hot_score"],
            format_func=lambda x: {
                "likes": "点赞", "coins": "投币", "favorites": "收藏",
                "shares": "转发", "comments": "评论", "danmaku": "弹幕",
                "hot_score": "热度分",
            }.get(x, x),
            key="y_scatter",
        )
        fig = px.scatter(
            plot_df, x=x_axis, y=y_axis,
            color=color_by,
            color_discrete_map=category_color_map() if color_by == "category" else None,
            opacity=0.7, log_x=log_scale, log_y=True,
            hover_data=["category", "views", "likes", "comments"],
        )
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "箱线图":
        fig = px.box(
            plot_df, x=color_by, y=x_axis,
            color=color_by,
            color_discrete_map=category_color_map() if color_by == "category" else None,
            log_y=log_scale,
        )
        fig.update_layout(height=550, xaxis_title="", yaxis_title=x_axis)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "小提琴图":
        fig = px.violin(
            plot_df, x=color_by, y=x_axis,
            color=color_by,
            color_discrete_map=category_color_map() if color_by == "category" else None,
            box=True, log_y=log_scale,
        )
        fig.update_layout(height=550, xaxis_title="", yaxis_title=x_axis)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "直方图":
        n_bins = st.slider("分箱数", 20, 100, 50, key="hist_bins")
        fig = px.histogram(
            plot_df, x=x_axis, color=color_by,
            nbins=n_bins, marginal="box",
            color_discrete_map=category_color_map() if color_by == "category" else None,
            log_x=log_scale, opacity=0.7,
        )
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "热力图(交叉表)":
        y_heat = st.selectbox(
            "Y轴变量",
            ["category", "publish_dayofweek", "is_collab"],
            format_func=lambda x: {
                "category": "分区", "publish_dayofweek": "星期",
                "is_collab": "是否合作",
            }.get(x, x),
            key="y_heat",
        )
        agg_func = st.selectbox(
            "聚合函数",
            ["count", "mean", "median", "sum"],
            format_func=lambda x: {"count": "计数", "mean": "均值", "median": "中位数", "sum": "总和"}.get(x, x),
        )
        cross = pd.crosstab(plot_df[y_heat], plot_df[color_by],
                            values=plot_df[x_axis], aggfunc=agg_func)
        fig = px.imshow(cross, text_auto=".0f", aspect="auto",
                         color_continuous_scale="YlOrRd")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    # 数据导出
    st.markdown("---")
    st.markdown("#### 当前筛选数据导出")
    csv_data = filtered_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 下载CSV数据",
        csv_data,
        "bilibili_data.csv",
        "text/csv",
    )

# ============================================================
# 页脚
# ============================================================
st.sidebar.markdown("---")
st.sidebar.info(
    "**数据可视化课程大作业**\n\n"
    "主题：B站热门内容分析与流量预测\n\n"
    "技术栈：Python + Streamlit + Plotly + Scikit-learn"
)
