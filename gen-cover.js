#!/usr/bin/env node
// 用法: node gen-cover.js "文章标题" "关键词描述" output.jpg
// 例如: node gen-cover.js "后端五件套" "FastAPI Docker PostgreSQL server tech" backend-stack.jpg

const https = require('https')
const http = require('http')
const fs = require('fs')
const path = require('path')
const url = require('url')

const API_KEY = process.env.GEMINI_API_KEY
if (!API_KEY) {
  console.error('请设置环境变量: export GEMINI_API_KEY=your_key')
  process.exit(1)
}
const API_URL = 'https://api.ikuncode.cc/v1beta/models/gemini-3.1-flash-image-preview:generateContent'

const [,, title, keywords, outputFile] = process.argv

if (!title) {
  console.log('用法: node gen-cover.js "文章标题" "关键词" output.jpg')
  process.exit(1)
}

const prompt = `Create a minimalist tech blog cover image for an article titled "${title}".
Style: dark background (#0a0a0a), flat design, geometric shapes, glowing accent colors (blue #38BDF8 or purple).
Theme keywords: ${keywords || title}.
No text in the image. Clean, modern, suitable for a developer blog.`

const body = JSON.stringify({
  contents: [{ parts: [{ text: prompt }] }],
  generationConfig: {
    responseModalities: ['IMAGE'],
    responseMimeType: 'image/jpeg'
  }
})

const parsed = new url.URL(API_URL)
const options = {
  hostname: parsed.hostname,
  path: parsed.pathname,
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'x-goog-api-key': API_KEY,
    'Content-Length': Buffer.byteLength(body)
  }
}

console.log(`生成封面: "${title}"...`)

const req = https.request(options, (res) => {
  let data = ''
  res.on('data', chunk => data += chunk)
  res.on('end', () => {
    try {
      const json = JSON.parse(data)
      const parts = json?.candidates?.[0]?.content?.parts
      const imgPart = parts?.find(p => p.inlineData)
      if (!imgPart) {
        console.error('未找到图片数据:', JSON.stringify(json).slice(0, 300))
        process.exit(1)
      }
      const imgData = Buffer.from(imgPart.inlineData.data, 'base64')
      const outPath = outputFile || `cover-${Date.now()}.jpg`
      fs.writeFileSync(outPath, imgData)
      console.log(`✓ 已保存: ${outPath} (${(imgData.length / 1024).toFixed(0)} KB)`)
    } catch (e) {
      console.error('解析失败:', e.message)
      console.error(data.slice(0, 500))
    }
  })
})

req.on('error', e => console.error('请求失败:', e.message))
req.write(body)
req.end()
