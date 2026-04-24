# A股量化分析 MVP

这是一个最快速可行的 A 股量化分析系统雏形，目标不是一开始就做成全自动交易，而是先完成 4 件事：

1. 获取 A 股行情数据
2. 扫描候选股票
3. 给出买卖时点信号
4. 对单只股票做简易回测
5. 用 Streamlit 可视化查看扫描结果和个股图表

当前策略采用趋势跟随思路：

- 均线多头排列：20 日均线 > 50 日均线 > 100 日均线
- 价格突破：收盘价突破过去 55 天高点
- 成交量确认：成交量大于 20 日均量的 1.5 倍
- RSI 仅作辅助观察，不再作为硬性入场过滤
- 卖出条件：跌破短均线、趋势破坏、突破失败快退或触发 ATR 止损

当前扫描池默认采用“综合活跃度抽样”，不是简单只看成交额前 N：

- 先用最低成交额保证流动性
- 再用最低换手率过滤不活跃标的
- 然后综合成交额、换手率、振幅和涨跌幅绝对值，抽取更有交易意义的前 80 只候选股

## 为什么这是最快方案

- Python 成熟，和 VS Code + Copilot 配合最好
- AkShare 对 A 股友好，上手快
- 历史日线增加 BaoStock 兜底，公开源抖动时更稳
- 不依赖券商 API，先把研究和信号跑通
- 先做日线级别系统，复杂度明显低于分时和实盘撮合

## 运行前提

- Windows 上先安装 Python 3.11 或 3.12
- 安装时勾选 `Add python.exe to PATH`
- VS Code 安装 Python 扩展

## 建议开发路径

### 第 1 阶段：先跑研究系统

完成下面两个命令即可：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
python -m ashare_quant.cli preheat --as-of 2026-04-02 --limit 20
python -m ashare_quant.cli scan
python -m ashare_quant.cli backtest 600519 --start 2022-01-01
& ".\.venv\Scripts\streamlit.exe" run streamlit_app.py
```

### 第 2 阶段：加每日提醒

每天定时运行扫描，把结果写入 CSV、企业微信、钉钉、邮箱或 Telegram。

### 第 3 阶段：再做策略迭代

可以继续加入：

- 财务因子
- 行业强度
- 北向资金
- 指数择时
- 组合仓位管理

## 目录说明

```text
config/universe.yaml        策略与市场过滤参数
src/ashare_quant/data.py    A 股数据获取与缓存
src/ashare_quant/strategy.py 指标、选股、买卖信号
src/ashare_quant/backtest.py 简易回测
src/ashare_quant/cli.py     命令行入口
streamlit_app.py            Streamlit 可视化看板
data/                       本地缓存和扫描结果
```

## 数据源说明

- 全市场快照优先使用 efinance，失败时回退到 AkShare 东方财富接口；如果两者都失败，则优先使用本地缓存的 `data/latest_snapshot.csv`，只在没有缓存时才尝试 AkShare 遗留接口
- 个股历史日线优先使用 Tushare，失败时再回退到 efinance、AkShare `stock_zh_a_daily`、AkShare `stock_zh_a_hist`，BaoStock 仅保留为最后兜底
- 基准指数历史默认优先使用 AkShare 指数接口，失败时再回退到 BaoStock 和本地缓存；只有在 `data_source.use_tushare_for_index: true` 时才会优先尝试 Tushare
- 这意味着单股分析、回测和基于已有快照的扫描，在公开源波动时更容易继续运行

### Tushare Token 配置

系统会按下面顺序读取 Tushare Token：

- 进程环境变量 `TUSHARE_TOKEN`
- `config/universe.yaml` 里的 `data_source.tushare_token`
- `config/universe.yaml` 指定的 `.env` 文件，默认读取项目根目录 `.env`

推荐做法是在项目根目录创建 `.env`：

```powershell
Copy-Item .env.example .env
```

然后填入：

```text
TUSHARE_TOKEN=你的_tushare_token
```

如果你不想使用 `.env`，也可以直接把 token 写到 `config/universe.yaml` 的 `data_source.tushare_token`。未配置 token 时，系统会自动跳过 Tushare，继续走旧的数据源回退链路。

## 当前交易语义

- 买点、补仓点、卖点都以当日收盘数据确认
- 回测默认按下一交易日开盘价执行，不再使用同日收盘成交的乐观口径
- 默认参数采用更偏专业的趋势跟随口径：20/50/100 均线、55 日突破、1.5 倍量能确认、2.5 x ATR 止损
- 默认扫描池采用 `composite` 综合活跃度模式：满足流动性后，优先分析更活跃的股票，而不是单纯按成交额从大到小截断
- 卖点除了趋势破坏外，还加入了“突破失败快退”：买点后 3 个交易日内重新跌回突破位下方且当日转弱，会直接提示离场
- 扫描结果增加市场宽度过滤：如果扫描池里处于多头趋势的股票占比不足，系统会把买入/补仓建议降级为观察，减少弱环境下追突破
- 当指数数据可用时，系统会优先结合基准指数趋势和市场宽度共同判断环境；当指数数据不可用时，会自动回退到宽度过滤
- 扫描表按执行优先级排序，并用颜色区分“快退 / 卖出 / 买入 / 补仓 / 观察”
- 市场环境现分为“风险开 / 中性 / 风险关”三档：中性环境下默认不新开仓，但允许结合原有持仓计划观察补仓机会
- 看板启动时会优先加载本地最近一次扫描结果，避免每次打开都等待全量刷新
- 全量扫描增加超时保护：超过设定时间会先展示已完成的部分结果，并保留旧的完整结果供快速查看
- 看板服务启动时会自动清理超过 `scan.history_days` 的本地历史缓存文件，默认只保留最近 260 个交易日窗口内仍有参考价值的数据
- 盘中提示层的阈值已参数化到 `config/universe.yaml` 的 `intraday_prompt` 段，可单独调整突破确认、最大追价、回踩容忍和盘中波动提示阈值

## 你接下来怎么用

### 扫描全市场候选股

如果你当前网络波动较大，建议先预热候选池历史缓存，再跑扫描：

```powershell
python -m ashare_quant.cli preheat --as-of 2026-04-02 --limit 80
```

这个命令会先按当前扫描池抽样规则选出候选股，并把它们的历史日线尽量预拉到本地缓存。预热完成后，再执行扫描通常更容易拿到完整结果。

```powershell
python -m ashare_quant.cli scan --top 20
```

默认启用严格收盘后模式：

- 如果请求日期是今天，但数据源最新完整日线还没到今天，系统会阻止输出次日建议
- 这种情况下会提示 `严格收盘后模式已启用...已阻止输出次日选股建议`

如果你只是想临时查看参考结果（不用于正式次日交易），可加：

```powershell
python -m ashare_quant.cli scan --allow-stale --top 20
```

输出字段里最重要的是：

- buy_signal: 是否满足买点
- sell_signal: 是否出现离场信号
- score: 综合强度评分

### 回测单只股票

```powershell
python -m ashare_quant.cli backtest 600519 --start 2021-01-01
```

### 打开可视化看板

```powershell
& ".\.venv\Scripts\streamlit.exe" run streamlit_app.py
```

看板里可以直接完成：

- 全市场扫描
- 查看候选股列表
- 绘制个股 K 线与均线
- 标记买卖点
- 展示单股回测结果

## 重要提醒

这个 MVP 适合做研究、选股和提示，不适合直接无审查实盘下单。A 股有涨跌停、停牌、滑点、手续费、复权处理、指数环境切换等真实问题，后续应逐步补上。

如果你要继续往可用产品推进，下一步最值得加的是：

1. 指数环境过滤，比如沪深 300 或中证 500 趋势为正时才开仓
2. 邮件或企业微信提醒
3. 多股票组合回测
4. 财务因子和行业强度过滤
