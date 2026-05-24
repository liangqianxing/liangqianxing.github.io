import 'dotenv/config'
import { createWriteStream } from 'node:fs'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { get } from 'node:https'
import { basename, extname } from 'node:path'
import { pipeline } from 'node:stream/promises'
import zlib from 'zlib'

const RANKING_URL = 'https://www.pixiv.net/ranking.php?mode=daily&content=illust&format=json&p=1'
const COVER_DIR = 'public/covers'
const COVERS_JSON = 'public/covers.json'
const MAX_COVERS = 24

function rankingHeaders(cookie: string): Record<string, string> {
  return {
    Cookie: cookie,
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    Referer: 'https://www.pixiv.net/ranking.php?mode=daily&content=illust',
    Accept: 'application/json, text/javascript, */*; q=0.01',
    'Accept-Language': 'zh-CN,zh;q=0.9,ja;q=0.8,en;q=0.7',
    'Accept-Encoding': 'gzip, deflate, br',
    'X-Requested-With': 'XMLHttpRequest',
    'sec-ch-ua': '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
    Connection: 'keep-alive',
  }
}

function imageHeaders(cookie: string): Record<string, string> {
  return {
    Referer: 'https://www.pixiv.net/',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    Cookie: cookie,
    Accept: 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,ja;q=0.8',
    'Sec-Fetch-Dest': 'image',
    'Sec-Fetch-Mode': 'no-cors',
    'Sec-Fetch-Site': 'cross-site',
  }
}

type PixivWork = { id?: number | string; url?: string; illust_id?: number | string }
type PixivRanking = { contents?: PixivWork[]; works?: Array<{ work?: PixivWork }> }

function requestText(url: string, headers: Record<string, string>): Promise<string> {
  return new Promise((resolve, reject) => {
    get(url, { headers }, (res) => {
      if (res.statusCode && res.statusCode >= 400) {
        res.resume()
        reject(new Error(`HTTP ${res.statusCode}`))
        return
      }

      const encoding = res.headers['content-encoding']
      let stream: NodeJS.ReadableStream = res

      if (encoding === 'gzip') {
        stream = res.pipe(zlib.createGunzip())
      } else if (encoding === 'br') {
        stream = res.pipe(zlib.createBrotliDecompress())
      } else if (encoding === 'deflate') {
        stream = res.pipe(zlib.createInflate())
      }

      let body = ''
      stream.on('data', (chunk: Buffer | string) => { body += chunk.toString() })
      stream.on('end', () => resolve(body))
      stream.on('error', reject)
    }).on('error', reject)
  })
}

function download(url: string, file: string, cookie: string): Promise<void> {
  return new Promise((resolve, reject) => {
    get(url, { headers: imageHeaders(cookie) }, async (res) => {
      if (res.statusCode && res.statusCode >= 400) {
        res.resume()
        reject(new Error(`HTTP ${res.statusCode}`))
        return
      }
      try {
        await pipeline(res, createWriteStream(file))
        resolve()
      } catch (error) {
        reject(error)
      }
    }).on('error', reject)
  })
}

async function readExistingCovers(): Promise<string[]> {
  try {
    const value = JSON.parse(await readFile(COVERS_JSON, 'utf8')) as unknown
    return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : []
  } catch {
    return []
  }
}

function extractWorks(data: PixivRanking): PixivWork[] {
  const works = data.contents ?? data.works?.map((item) => item.work).filter((item): item is PixivWork => Boolean(item)) ?? []
  return works.filter((work) => typeof work.url === 'string')
}

function coverName(work: PixivWork, index: number): string {
  const source = work.url ?? ''
  const extension = extname(new URL(source).pathname) || '.jpg'
  const id = String(work.id ?? work.illust_id ?? basename(source, extension) ?? index)
  return `${id}${extension}`.replace(/[^a-zA-Z0-9._-]/g, '-')
}

async function main(): Promise<void> {
  const cookie = process.env.PIXIV_COOKIE
  if (!cookie) {
    console.error('PIXIV_COOKIE is missing; skipping cover fetch.')
    return
  }

  await mkdir(COVER_DIR, { recursive: true })

  let works: PixivWork[]
  try {
    const text = await requestText(RANKING_URL, rankingHeaders(cookie))
    const ranking = extractWorks(JSON.parse(text) as PixivRanking)
    console.log(`Got ${ranking.length} items from ranking`)
    works = ranking.slice(0, MAX_COVERS)
  } catch (error) {
    console.warn(`Could not fetch Pixiv ranking; keeping existing covers. ${error instanceof Error ? error.message : String(error)}`)
    return
  }

  const covers: string[] = []
  for (const [index, work] of works.entries()) {
    if (!work.url) continue
    let name = `cover-${index}.jpg`
    try {
      name = coverName(work, index)
      await download(work.url, `${COVER_DIR}/${name}`, cookie)
      covers.push(`/covers/${name}`)
    } catch (error) {
      console.warn(`Skipping cover ${name}: ${error instanceof Error ? error.message : String(error)}`)
    }
  }

  if (covers.length > 0) {
    await writeFile(COVERS_JSON, `${JSON.stringify(covers, null, 2)}
`, 'utf8')
    console.log(`Fetched ${covers.length} cover(s).`)
  } else {
    const existing = await readExistingCovers()
    console.warn(`No covers downloaded; keeping ${existing.length} existing cover(s).`)
  }
}

main().catch((error: unknown) => {
  console.warn(`Cover fetch skipped: ${error instanceof Error ? error.message : String(error)}`)
})
