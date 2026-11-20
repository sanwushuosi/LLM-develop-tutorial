# 质检小结：v2_X01_boiler / kept / n=40 / seed=1

结果：good 32 条，bad 8 条（20%）。命中次数：W03 ×6，W01 ×1，W05 ×1，W07 ×1，G01 ×1（2 条样本各命中两条规则）。
G02 / G03 / W02 / W04 / W06 / S01 / L01 均未命中：乱码、机械重复、HTML/JS 残留、语言异常在这一批里已经清理干净。

- **最常见的问题是“正文没抓到，只剩全站模板”（W03，6/8 条 bad）。** 这几条不是正文里夹了几行导航，而是页面主体本来就空了或很薄（结账页、空分类页、车行询价表单、号码查询页、只有一句描述的商品页、指标详情页），抽取器于是把 cookie 横幅、包邮/退货/先买后付卖点、表单同意声明、栏目导语卡片当成了正文。共同特征是**短**：去掉模板句后几乎什么都不剩。
  建议规则：维护一份模板句表（cookie 告知、`By clicking ...I consent`、`Free shipping ... orders over`、`return guarantee`、`Buy now, pay later`、`Visit Us In Store` 等），**先逐行删掉命中的模板行，再算剩余正文**；剩余少于 200 字符或少于 3 个完整句就丢弃整页。

- **整页丢弃信号：URL 与空页提示词。** 4 条 bad 光看 URL 或一句固定提示就能定：`forsale.godaddy.com` / `utm_medium=parkedpages`（域名出售，W07）、`/checkout/`（结账页）、`/search/?number=` + `Sign in to unlock`（登录墙，W01）、`No Product found in this category!`（空分类，G01）。
  建议规则：URL 黑名单（`/checkout`、`/cart`、`/my-account`、`/search?`、`forsale.`、`parkedpages`），再加上正文出现 `No Product(s) found` / `Sign in to unlock` / `The domain name ... is for sale` 时丢弃整页。这一类成本低、误杀少，可以直接上线。

- **列表/分类页只剩截断标题（W05）。** 例：`NewsInfoCategory?categoryId=` 页面只有 4~5 行标题和一段被截断的摘要。
  建议规则：URL 含 `Category` / `list` / `tag`，且正文行数 ≤ 6、多数行没有句末标点（或末行被截断）时丢弃。注意 `product-tag` 页里也有完整短公告（sample 031 判 good），所以不能只凭 URL 判，必须叠加“无完整句”这个条件。

- **站内重复的导语/卡片要靠跨页统计。** Cancer Atlas 指标页（sample 022）四句话通顺，但都是全站栏目导语，和 URL 指向的具体指标无关。单页规则抓不到，建议按域名统计句子频次：同一句在同域名下 ≥ N 个页面出现就删掉，删完为空的页面丢弃。这条是本批最拿不准的 bad，另一位质检员也可能判 good。

- **判 good 但仍值得留意的模式（可作为 X 规则候选）：** 商品/二手设备页带大段营销与合规模板（sample 032）、首页带大量 Google 评论拼接和空白缩进行（sample 006）、正文首行是 cookie 横幅（sample 020）、结尾挂着 `Login to ask a question`/`No Reviews Submitted` 之类（sample 000、010）。这些按底线口径都算轻问题，判 good；但可以考虑补一条 **X01“电商/服务页模板句占比过高”**：模板句占全文字符 > 40% 即判 bad，同时用行级删除把首尾的 cookie 和评论区提示清掉。
