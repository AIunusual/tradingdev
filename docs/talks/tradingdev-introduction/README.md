# TradingDev 簡報編輯指南

簡報內容與講者稿維護在 [slides.md](slides.md)，版型維護在 [theme.css](theme.css)；
[index.html](index.html) 是產生後提交與發布的瀏覽器版本。
大綱、章節整理與後續目標集中放在 `slides.md`，本文件只說明編輯與發布操作。

## 編輯與預覽

首次使用先依[開發指南](../../development.md#簡報的編輯與輸出)初始化環境與 Marp 工具鏈，
再用 VS Code 開啟整個 repository 資料夾。

1. 開啟命令選擇區，執行 **Tasks: Run Task → slides: watch**。
2. 編輯並儲存 `slides.md` 或 `theme.css`，工作會自動更新同目錄的 `index.html`。
3. 用瀏覽器開啟 `index.html`；每次儲存後重新整理瀏覽器查看結果。
4. 結束時執行 **Tasks: Terminate Task → slides: watch**。

編譯訊息可在 `slides: watch` 的專用終端機查看。只需單次輸出時，執行
**Tasks: Run Task → slides: build**。自動啟動監看、命令列操作與本機 HTTP
預覽方式見[開發指南](../../development.md#簡報的編輯與輸出)。
提交簡報內容或版型修改時，一併提交重新產生的 `index.html`。

瀏覽器版內含樣式與播放程式，可離線開啟。使用左右方向鍵換頁，`F` 切換全螢幕，
`P` 開啟講者模式；講者模式須允許新視窗，本機檔案模式有通訊限制時改用 HTTP 預覽。

## 章節與子主題版型

- **章首頁**使用 `chapter`：深色底、章節編號與主題說明。
- **子主題起始頁**使用 `subsection`：沿用淺色底，標題上方以綠色短線標示轉場。
  頁首用 `header` 標出章節與子主題，例如 `02　策略研究　／　02.2　策略實驗與結果`。
- **延伸內容頁**沿用相同 `header`，不加 `subsection`，依內容選用表格、列表或引言。
  下一個子主題或章節開始時更新 `header`，小結頁則改回章節名稱。

子主題起始頁同時呈現該主題的第一個重點，不必額外插入分隔頁。概念引言可搭配
`statement`，如「這裡的『研究能力』」與「自己的規則也能成為策略」；表格較密時可搭配
`compact`。這兩個 class 調整內容呈現，不代表章節層級，`statement` 也可用於小結。

只有需要區分的主題群才設子主題，連續的短章節可直接展開。後續開發方向沿用
`TODO-01` 等既有識別碼作為子主題標示；同一目標的補充頁沿用其頁首。
新增或移動頁面時，同步核對閱讀路線及其他 `index.html#頁碼` 連結。

## 在 GitHub 網頁上播放

GitHub 的一般檔案頁只會顯示 HTML 原始碼。若要直接播放，可使用同一 repository
的 GitHub Pages。

取得發布授權並透過 PR 合併至主線後，由管理者在
**Settings → Pages → Deploy from a branch** 選擇主線與 `/docs`。
這會發布 `/docs` 下的內容，`.nojekyll` 讓 GitHub 直接提供靜態檔案；
已有 Pages 網站時，先確認原有發布設定。

沒有自訂網域且使用目前 repository 名稱時，預期位址為：

```text
https://kuan-lun.github.io/tradingdev/talks/tradingdev-introduction/
```

此位址不代表已發布；完整設定見
[GitHub Pages 官方說明](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)。
