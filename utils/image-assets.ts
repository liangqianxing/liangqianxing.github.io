import assets from '~/data/image-assets.json'

interface ImageAsset {
  width: number
  height: number
  variants: { src: string; width: number }[]
}

export function getImageAsset(src: string): ImageAsset | undefined {
  return (assets as Record<string, ImageAsset>)[src.split(/[?#]/)[0]]
}

export function imageDimensions(src: string, width?: number | string, height?: number | string) {
  const asset = getImageAsset(src)
  if (!asset) return { width, height }
  if (width != null && height == null && Number(width) > 0) {
    return { width, height: Math.round(Number(width) * asset.height / asset.width) }
  }
  if (height != null && width == null && Number(height) > 0) {
    return { width: Math.round(Number(height) * asset.width / asset.height), height }
  }
  return { width: width ?? asset.width, height: height ?? asset.height }
}

export function imageWithBase(src: string, baseURL: string) {
  if (!src.startsWith('/') || src.startsWith('//') || baseURL === '/') return src
  const base = `/${baseURL.replace(/^\/+|\/+$/g, '')}/`
  return src.startsWith(base) ? src : `${base}${src.slice(1)}`
}
