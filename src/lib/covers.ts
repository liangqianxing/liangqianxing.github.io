import covers from '../../public/covers.json';

const list = covers as string[];

export function coverFor(seed: string): string | undefined {
  if (list.length === 0) return undefined;
  let h = 2166136261;
  for (let i = 0; i < seed.length; i++) {
    h ^= seed.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return list[Math.abs(h) % list.length];
}

export function coverAt(index: number): string | undefined {
  if (list.length === 0) return undefined;
  return list[((index % list.length) + list.length) % list.length];
}

export const hasCovers = list.length > 0;
