import datetime
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests
import yfinance as yf

# Backup list in case HKEX website is temporarily unreachable
FALLBACK_HKEX_LIST = [
    "0001.HK", "0002.HK", "0003.HK", "0005.HK", "0006.HK", "0011.HK", "0012.HK",
    "0016.HK", "0017.HK", "0027.HK", "0066.HK", "0101.HK", "0175.HK", "0267.HK",
    "0288.HK", "0291.HK", "0316.HK", "0386.HK", "0388.HK", "0669.HK", "0700.HK",
    "0762.HK", "0857.HK", "0883.HK", "0939.HK", "0941.HK", "0960.HK", "0968.HK",
    "0981.HK", "0992.HK", "1024.HK", "1038.HK", "1088.HK", "1093.HK", "1109.HK",
    "1113.HK", "1177.HK", "1211.HK", "1299.HK", "1378.HK", "1398.HK", "1810.HK",
    "1928.HK", "1929.HK", "2015.HK", "2020.HK", "2269.HK", "2313.HK", "2318.HK",
    "2319.HK", "2331.HK", "2382.HK", "2388.HK", "2600.HK", "2688.HK", "3690.HK",
    "3968.HK", "3988.HK", "6618.HK", "9618.HK", "9866.HK", "9868.HK", "9888.HK", "9988.HK"
]

def fetch_all_hkex_equities():
    """
    Dynamically fetches the latest official List of Securities from HKEX.
    Filters exclusively for Listed Equities (Main Board & GEM Ordinary Shares).
    Excludes Warrants, CBBCs, Debt, ETFs, REITs, and Structured Products.
    """
    print("Fetching official HKEX securities list from exchange...")
    url = "https://www.hkex.com.hk/eng/services/trading/securities/securitieslists/ListOfSecurities.xlsx"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()

        # Load Excel bytes into pandas dataframe
        df = pd.read_excel(io.BytesIO(response.content), skiprows=2, engine="openpyxl")
        df.columns = [str(col).strip() for col in df.columns]

        cat_col = next((c for c in df.columns if "category" in c.lower() and "sub" not in c.lower()), None)
        subcat_col = next((c for c in df.columns if "sub-category" in c.lower() or "subcategory" in c.lower()), None)
        code_col = next((c for c in df.columns if "stock code" in c.lower() or "code" in c.lower()), None)
        name_col = next((c for c in df.columns if "name" in c.lower()), None)

        if not code_col:
            raise ValueError("Stock Code column not found in HKEX file.")

        equity_mask = pd.Series(True, index=df.index)
        if cat_col:
            equity_mask &= df[cat_col].astype(str).str.contains("Equity", case=False, na=False)
        if subcat_col:
            equity_mask &= df[subcat_col].astype(str).str.contains("Equity|Ordinary Shares|Shares", case=False, na=False)

        exclude_patterns = r"Warrant|CBBC|Debt|Bond|Fund|ETF|ETP|REIT|Trust|Preference|Inline"
        exclude_mask = pd.Series(False, index=df.index)
        if cat_col:
            exclude_mask |= df[cat_col].astype(str).str.contains(exclude_patterns, case=False, na=False)
        if name_col:
            exclude_mask |= df[name_col].astype(str).str.contains(r"ETF|REIT|BOND|CALL|PUT|CW|PW|VT|N2|R2", case=False, na=False)

        filtered_df = df[equity_mask & (~exclude_mask)].copy()

        tickers = []
        for raw_code in filtered_df[code_col]:
            try:
                clean_num = int(raw_code)
                code_str = str(clean_num)
                formatted_ticker = f"{code_str.zfill(4)}.HK" if len(code_str) <= 4 else f"{code_str.zfill(5)}.HK"
                tickers.append(formatted_ticker)
            except (ValueError, TypeError):
                continue

        tickers = list(dict.fromkeys(tickers))

        if len(tickers) > 100:
            print(f"Successfully loaded {len(tickers)} HKEX equity stocks.")
            return tickers

    except Exception as e:
        print(f"⚠️ Dynamic HKEX fetch failed ({e}). Using fallback list.")

    return FALLBACK_HKEX_LIST

def analyze_stock(symbol):
    """
    Calculates technical indicators for a given HK stock ticker:
    - Stage 1: Close > MA60 & MA60 sloping up & 3M Return > 0
    - Stage 2: Stage 1 + 5D Avg Vol >= 1.2x 20D Avg Vol + 5D Return > 0
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1y")

        if df.empty or len(df) < 65:
            return None

        df["MA60"] = df["Close"].rolling(window=60).mean()

        latest_close = float(df["Close"].iloc[-1])
        latest_ma60 = float(df["MA60"].iloc[-1])
        prev_ma60 = float(df["MA60"].iloc[-5])

        if pd.isna(latest_ma60) or pd.isna(prev_ma60) or latest_close <= 0:
            return None

        price_above_ma60 = latest_close > latest_ma60
        ma60_sloping_up = latest_ma60 > prev_ma60

        if not (price_above_ma60 and ma60_sloping_up):
            return None

        close_3m_ago = float(df["Close"].iloc[-63]) if len(df) >= 63 else float(df["Close"].iloc[0])
        m3_return = ((latest_close - close_3m_ago) / close_3m_ago) * 100

        if m3_return < 0:
            return None

        close_5d_ago = float(df["Close"].iloc[-6]) if len(df) >= 6 else float(df["Close"].iloc[0])
        m5d_return = ((latest_close - close_5d_ago) / close_5d_ago) * 100

        vol_5d_avg = float(df["Volume"].iloc[-5:].mean())
        vol_20d_avg = float(df["Volume"].iloc[-20:].mean())
        vol_ratio = (vol_5d_avg / vol_20d_avg) if vol_20d_avg > 0 else 0.0

        stage2_pass = (vol_ratio >= 1.2) and (m5d_return > 0)

        info = ticker.info or {}
        short_name = info.get("shortName") or info.get("longName") or symbol

        clean_code = symbol.replace(".HK", "").lstrip("0")
        if not clean_code:
            clean_code = "0"

        return {
            "symbol": symbol,
            "clean_code": clean_code,
            "name": short_name,
            "close": round(latest_close, 2),
            "ma60": round(latest_ma60, 2),
            "dist_ma60": round(((latest_close - latest_ma60) / latest_ma60) * 100, 2),
            "m3_return": round(m3_return, 2),
            "m5d_return": round(m5d_return, 2),
            "vol_ratio": round(vol_ratio, 2),
            "stage2_pass": stage2_pass,
        }

    except Exception:
        return None

def run_screener():
    watchlist = fetch_all_hkex_equities()
    total_stocks = len(watchlist)
    print(f"\nStarting parallel technical screening on {total_stocks} HKEX equity stocks...")

    stage1_list = []
    stage2_list = []

    max_workers = 16
    completed = 0

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {executor.submit(analyze_stock, sym): sym for sym in watchlist}

        for future in as_completed(future_to_symbol):
            completed += 1
            if completed % 200 == 0 or completed == total_stocks:
                print(f"Progress: [{completed}/{total_stocks}] stocks analyzed ({(completed/total_stocks)*100:.1f}%)...")

            res = future.result()
            if res:
                stage1_list.append(res)
                if res["stage2_pass"]:
                    stage2_list.append(res)

    elapsed = round(time.time() - start_time, 2)
    print(f"\nScreening finished in {elapsed}s. Total Analyzed: {total_stocks} | Stage 1: {len(stage1_list)} | Stage 2: {len(stage2_list)}")

    stage1_list.sort(key=lambda x: x["vol_ratio"], reverse=True)
    stage2_list.sort(key=lambda x: x["vol_ratio"], reverse=True)

    return stage1_list, stage2_list, total_stocks

def generate_html_report(stage1_results, stage2_results, total_scanned):
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_str = datetime.date.today().strftime("%Y-%m-%d")

    def build_rows(items, is_stage2=False):
        if not items:
            return '<tr><td colspan="8" class="p-8 text-center text-slate-400 font-medium">本日無符合篩選條件之個股</td></tr>'

        html_snippets = []
        for stock in items:
            ret_color = "text-emerald-600 font-bold" if stock["m5d_return"] > 0 else "text-rose-600 font-bold"
            vol_badge = f'<span class="px-2.5 py-1 rounded-full text-xs font-bold bg-purple-100 text-purple-700">{stock["vol_ratio"]}x</span>' if is_stage2 else f'{stock["vol_ratio"]}x'

            tv_url = f"https://www.tradingview.com/chart/?symbol=HKEX:{stock['clean_code']}"
            futu_url = f"https://www.futunn.com/stock/{stock['clean_code']}-HK"

            row = f"""
            <tr class="hover:bg-indigo-50/40 transition border-b border-slate-100 stock-row">
                <td class="p-4 font-bold text-slate-900 symbol-col">{stock['symbol']}</td>
                <td class="p-4 font-medium text-slate-800 name-col">{stock['name']}</td>
                <td class="p-4 font-semibold text-slate-900">${stock['close']}</td>
                <td class="p-4 text-slate-600">${stock['ma60']} <span class="text-xs text-emerald-600 font-medium">(+{stock['dist_ma60']}%)</span></td>
                <td class="p-4 text-emerald-600 font-semibold">+{stock['m3_return']}%</td>
                <td class="p-4 {ret_color}">{stock['m5d_return']}%</td>
                <td class="p-4 font-semibold">{vol_badge}</td>
                <td class="p-4 flex items-center gap-2">
                    <a href="{tv_url}" target="_blank" class="px-2.5 py-1.5 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg text-xs font-semibold shadow-sm transition inline-flex items-center gap-1">
                        📊 TradingView
                    </a>
                    <a href="{futu_url}" target="_blank" class="px-2.5 py-1.5 bg-amber-500 hover:bg-amber-600 text-white rounded-lg text-xs font-semibold shadow-sm transition inline-flex items-center gap-1">
                        🐮 富途
                    </a>
                </td>
            </tr>
            """
            html_snippets.append(row)
        return "".join(html_snippets)

    html_content = f"""<!DOCTYPE html>
<html lang="zh-HK">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>港股全市場自動選股儀表板 | HKEX Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Plus Jakarta Sans', sans-serif; }}
        .canva-card {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(226, 232, 240, 0.8);
        }}
    </style>
</head>
<body class="bg-slate-100 text-slate-800 min-h-screen p-4 md:p-8">
    <div class="max-w-7xl mx-auto space-y-6">
        
        <!-- Header Banner -->
        <div class="bg-gradient-to-r from-indigo-900 via-purple-900 to-slate-900 text-white p-6 md:p-8 rounded-3xl shadow-xl flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
            <div>
                <div class="flex items-center gap-2">
                    <span class="px-3 py-1 bg-indigo-500/30 text-indigo-200 rounded-full text-xs font-semibold uppercase tracking-wider">Automated HKEX Full Market Analytics</span>
                    <span class="px-3 py-1 bg-emerald-500/30 text-emerald-200 rounded-full text-xs font-semibold">Excludes Warrants & CBBCs</span>
                </div>
                <h1 class="text-2xl md:text-3xl font-extrabold mt-2">📈 港股全市場自動選股儀表板</h1>
                <p class="text-sm text-slate-300 mt-1">每日港股收盤自動掃描香港交易所全數正股（主板 + GEM）| 季線 (MA60) 向上 + 量能爆發雙階篩選器</p>
            </div>
            <div class="bg-white/10 backdrop-blur-md px-4 py-3 rounded-2xl border border-white/20 text-right">
                <p class="text-xs text-slate-300">數據更新時間 (HKT)</p>
                <p class="text-sm font-bold text-indigo-200 mt-0.5">{now_str}</p>
            </div>
        </div>

        <!-- Metrics Overview -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div class="canva-card p-6 rounded-2xl shadow-sm">
                <p class="text-xs font-bold text-slate-400 uppercase tracking-wider">全港股正股掃描總數</p>
                <p class="text-3xl font-extrabold text-slate-800 mt-2">{total_scanned} <span class="text-xs font-normal text-slate-500">檔股票</span></p>
            </div>
            <div class="canva-card p-6 rounded-2xl shadow-sm border-l-4 border-indigo-500">
                <p class="text-xs font-bold text-indigo-600 uppercase tracking-wider">階段一符合 (站上季線+動能向上)</p>
                <p class="text-3xl font-extrabold text-indigo-600 mt-2">{len(stage1_results)} <span class="text-xs font-normal text-slate-500">檔</span></p>
            </div>
            <div class="canva-card p-6 rounded-2xl shadow-sm border-l-4 border-purple-500">
                <p class="text-xs font-bold text-purple-600 uppercase tracking-wider">階段二精選 (主力爆量 > 1.2x)</p>
                <p class="text-3xl font-extrabold text-purple-600 mt-2">{len(stage2_results)} <span class="text-xs font-normal text-slate-500">檔</span></p>
            </div>
        </div>

        <!-- Search Filter -->
        <div class="canva-card p-4 rounded-2xl shadow-sm flex flex-col md:flex-row items-center justify-between gap-4">
            <div class="w-full md:w-96 relative">
                <input type="text" id="searchInput" onkeyup="filterTables()" placeholder="🔍 搜尋股票代碼或名稱 (如 700 或 騰訊)..." class="w-full px-4 py-2.5 text-sm rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500 bg-white">
            </div>
            <p class="text-xs text-slate-400">已自動剔除衍生權證、牛熊證、債券、ETF及基金類別</p>
        </div>

        <!-- Stage 2 Table -->
        <div class="canva-card rounded-2xl shadow-sm overflow-hidden">
            <div class="p-6 border-b border-slate-100 bg-purple-50/50 flex justify-between items-center">
                <div>
                    <h2 class="text-lg font-bold text-purple-950 flex items-center gap-2">
                        🔥 階段二：精選主力爆量轉強個股 (優先關注)
                    </h2>
                    <p class="text-xs text-slate-500 mt-0.5">條件：站上向上季線 + 近3月動能正向 + 5日價格增長 + 5日均量放量達20日均量 1.2 倍以上</p>
                </div>
                <span class="px-3 py-1 bg-purple-100 text-purple-700 text-xs font-bold rounded-full">High Volume Momentum</span>
            </div>
            <div class="overflow-x-auto">
                <table class="w-full text-left border-collapse text-sm" id="stage2Table">
                    <thead>
                        <tr class="bg-slate-50 text-slate-500 font-bold border-b border-slate-100 text-xs uppercase">
                            <th class="p-4">代碼</th>
                            <th class="p-4">名稱</th>
                            <th class="p-4">最新價</th>
                            <th class="p-4">季線 (MA60)</th>
                            <th class="p-4">近3月漲幅</th>
                            <th class="p-4">5日漲幅</th>
                            <th class="p-4">量能放大比</th>
                            <th class="p-4">圖表快捷分析</th>
                        </tr>
                    </thead>
                    <tbody>
                        {build_rows(stage2_results, is_stage2=True)}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Stage 1 Table -->
        <div class="canva-card rounded-2xl shadow-sm overflow-hidden">
            <div class="p-6 border-b border-slate-100 bg-indigo-50/50">
                <h2 class="text-lg font-bold text-indigo-950 flex items-center gap-2">
                    📊 階段一：全市場趨勢轉強個股清單
                </h2>
                <p class="text-xs text-slate-500 mt-0.5">條件：股價高於60日季線且季線方向向上，近3個月具備正向累積漲幅</p>
            </div>
            <div class="overflow-x-auto">
                <table class="w-full text-left border-collapse text-sm" id="stage1Table">
                    <thead>
                        <tr class="bg-slate-50 text-slate-500 font-bold border-b border-slate-100 text-xs uppercase">
                            <th class="p-4">代碼</th>
                            <th class="p-4">名稱</th>
                            <th class="p-4">最新價</th>
                            <th class="p-4">季線 (MA60)</th>
                            <th class="p-4">近3月漲幅</th>
                            <th class="p-4">5日漲幅</th>
                            <th class="p-4">量能放大比</th>
                            <th class="p-4">圖表快捷分析</th>
                        </tr>
                    </thead>
                    <tbody>
                        {build_rows(stage1_results, is_stage2=False)}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Footer -->
        <footer class="text-center text-xs text-slate-400 py-4">
            自動選股機器人生成於 {date_str} | 全自動掃描 HKEX 正股清單 | 本網頁數據僅供技術分析與策略研究參考，不構成投資建議。
        </footer>

    </div>

    <script>
        function filterTables() {{
            const filter = document.getElementById('searchInput').value.toUpperCase();
            const rows = document.querySelectorAll('.stock-row');
            
            rows.forEach(row => {{
                const symbol = row.querySelector('.symbol-col')?.textContent || '';
                const name = row.querySelector('.name-col')?.textContent || '';
                if (symbol.toUpperCase().indexOf(filter) > -1 || name.toUpperCase().indexOf(filter) > -1) {{
                    row.style.display = "";
                }} else {{
                    row.style.display = "none";
                }}
            }});
        }}
    </script>
</body>
</html>
"""
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print("Successfully generated index.html dashboard!")

if __name__ == "__main__":
    print(f"[{datetime.datetime.now()}] Starting Full-Market HKEX Screening Job...")
    stage1, stage2, total_scanned = run_screener()
    generate_html_report(stage1, stage2, total_scanned)
