import { createHash } from "node:crypto";
import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import matter from "gray-matter";
import hljs from "highlight.js";
import { marked } from "marked";
import markedKatex from "marked-katex-extension";
import { markedHighlight } from "marked-highlight";

const baseUrl = (process.env.HALO_BASE_URL || "https://diycv.top").replace(/\/$/, "");
const token = process.env.HALO_TOKEN;
const policyName = process.env.HALO_POLICY_NAME || "default-policy";
const rootDir = process.cwd();
const postsDir = path.join(rootDir, "content", "posts");
const publicDir = path.join(rootDir, "public");

if (!token) {
  throw new Error("HALO_TOKEN is required");
}

marked.use(
  markedHighlight({
    emptyLangClass: "hljs",
    langPrefix: "hljs language-",
    highlight(code, language) {
      const normalized = language && hljs.getLanguage(language) ? language : "plaintext";
      return hljs.highlight(code, { language: normalized }).value;
    },
  }),
  markedKatex({ nonStandard: true, output: "mathml", throwOnError: false }),
);
marked.setOptions({ gfm: true });

async function haloRequest(apiPath, options = {}) {
  const response = await fetch(`${baseUrl}${apiPath}`, {
    ...options,
    headers: {
      Authorization: `Bearer ${token}`,
      ...(options.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...options.headers,
    },
  });

  const text = await response.text();
  if (!response.ok) {
    throw new Error(`${options.method || "GET"} ${apiPath} failed (${response.status}): ${text}`);
  }
  return text ? JSON.parse(text) : undefined;
}

async function listAll(apiPath) {
  const items = [];
  const separator = apiPath.includes("?") ? "&" : "?";

  for (let page = 1; ; page += 1) {
    const result = await haloRequest(`${apiPath}${separator}page=${page}&size=100`);
    items.push(...(result.items || []));
    if (!result.hasNext) break;
  }

  return items;
}

function stableName(prefix, value) {
  return `${prefix}-${createHash("sha256").update(value).digest("hex").slice(0, 16)}`;
}

function taxonSlug(prefix, value) {
  const ascii = value
    .normalize("NFKD")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
  return ascii || stableName(prefix, value);
}

function asList(value) {
  if (Array.isArray(value)) return value.filter(Boolean).map(String);
  return value ? [String(value)] : [];
}

function sourcePublishTime(frontMatter) {
  const match = frontMatter.match(/^date:\s*["']?([^"'\n]+)["']?\s*$/m);
  if (!match) return new Date().toISOString();
  const value = match[1].trim();
  const normalized = value.replace(" ", "T");
  const zoned = /^\d{4}-\d{2}-\d{2}$/.test(value)
    ? `${value}T00:00:00+08:00`
    : /(?:Z|[+-]\d{2}:?\d{2})$/i.test(normalized)
      ? normalized
      : `${normalized}+08:00`;
  const date = new Date(zoned);
  if (Number.isNaN(date.getTime())) throw new Error(`Invalid date: ${value}`);
  return date.toISOString();
}

function imageReferences(markdown) {
  const markdownImages = [...markdown.matchAll(/!\[[^\]]*\]\((\/images\/[A-Za-z0-9_./-]+\.(?:avif|gif|ico|jpe?g|png|svg|webp))\)/gi)].map(
    (match) => match[1],
  );
  const htmlImages = [...markdown.matchAll(/<img[^>]+src=["'](\/images\/[A-Za-z0-9_./-]+\.(?:avif|gif|ico|jpe?g|png|svg|webp))["']/gi)].map(
    (match) => match[1],
  );
  return [...markdownImages, ...htmlImages];
}

function mediaType(filename) {
  const extension = path.extname(filename).toLowerCase();
  return {
    ".avif": "image/avif",
    ".gif": "image/gif",
    ".ico": "image/x-icon",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".webp": "image/webp",
  }[extension] || "application/octet-stream";
}

async function uploadImages(posts, attachmentItems) {
  const attachments = attachmentItems.map((item) => item.attachment || item);
  const byDisplayName = new Map(attachments.map((item) => [item.spec.displayName, item]));
  const references = [
    ...new Set(
      posts.flatMap((post) => [
        ...imageReferences(post.body),
        ...(typeof post.data.cover === "string" && post.data.cover.startsWith("/images/")
          ? [post.data.cover]
          : []),
      ]),
    ),
  ].sort();
  const referencesByFilename = Map.groupBy(references, (reference) => path.basename(reference));
  const duplicateFilenames = [...referencesByFilename.entries()].filter(([, values]) => values.length > 1);
  if (duplicateFilenames.length) {
    const details = duplicateFilenames
      .map(([filename, values]) => `${filename}: ${values.join(", ")}`)
      .join("; ");
    throw new Error(`Halo attachment filenames must be unique: ${details}`);
  }
  const urls = new Map();

  for (const reference of references) {
    const relativePath = reference.replace(/^\//, "");
    const localPath = path.resolve(publicDir, relativePath);
    if (!localPath.startsWith(`${publicDir}${path.sep}`)) {
      throw new Error(`Image path escapes public directory: ${reference}`);
    }

    const displayName = path.basename(localPath);
    let attachment = byDisplayName.get(displayName);
    if (!attachment) {
      const bytes = await readFile(localPath);
      const form = new FormData();
      form.append("policyName", policyName);
      form.append("file", new Blob([bytes], { type: mediaType(localPath) }), displayName);
      attachment = await haloRequest(
        "/apis/api.console.halo.run/v1alpha1/attachments/upload?waitForPermalink=true",
        { method: "POST", body: form },
      );
      byDisplayName.set(displayName, attachment);
      process.stdout.write(`Uploaded ${reference}\n`);
    }

    const permalink = attachment.status?.permalink;
    if (!permalink) throw new Error(`Attachment has no permalink: ${displayName}`);
    urls.set(reference, permalink);
  }

  return urls;
}

async function ensureTaxonomy(apiPath, kind, prefix, displayNames, existingItems) {
  const byDisplayName = new Map(existingItems.map((item) => [item.spec.displayName, item]));
  const names = new Map();

  for (const displayName of [...new Set(displayNames)].sort()) {
    let item = byDisplayName.get(displayName);
    if (!item) {
      const isCategory = kind === "Category";
      item = await haloRequest(apiPath, {
        method: "POST",
        body: JSON.stringify({
          apiVersion: "content.halo.run/v1alpha1",
          kind,
          metadata: { name: stableName(prefix, displayName) },
          spec: {
            displayName,
            slug: taxonSlug(prefix, displayName),
            ...(isCategory
              ? {
                  children: [],
                  description: "",
                  hideFromList: false,
                  preventParentPostCascadeQuery: false,
                  priority: 0,
                }
              : {}),
          },
        }),
      });
      byDisplayName.set(displayName, item);
      process.stdout.write(`Created ${kind.toLowerCase()} ${displayName}\n`);
    }
    names.set(displayName, item.metadata.name);
  }

  return names;
}

function replaceContentLinks(markdown, imageUrls) {
  let result = markdown.replace(/\]\(\/posts\/([^)]+)\)/g, "](/archives/$1)");
  for (const [source, destination] of imageUrls) {
    result = result.split(source).join(destination);
  }
  return result;
}

async function main() {
  const postFiles = (await readdir(postsDir)).filter((name) => name.endsWith(".md")).sort();
  const posts = await Promise.all(
    postFiles.map(async (filename) => {
      const sourcePath = path.join(postsDir, filename);
      const source = await readFile(sourcePath, "utf8");
      const parsed = matter(source);
      return {
        body: parsed.content,
        data: parsed.data,
        filename,
        frontMatter: parsed.matter,
        slug: filename.replace(/\.md$/, ""),
      };
    }),
  );

  const [attachmentItems, existingCategories, existingTags, existingPosts] = await Promise.all([
    listAll("/apis/api.console.halo.run/v1alpha1/attachments"),
    listAll("/apis/content.halo.run/v1alpha1/categories"),
    listAll("/apis/content.halo.run/v1alpha1/tags"),
    listAll("/apis/content.halo.run/v1alpha1/posts"),
  ]);

  const imageUrls = await uploadImages(posts, attachmentItems);
  const categoryNames = posts.flatMap((post) => asList(post.data.categories));
  const tagNames = posts.flatMap((post) => asList(post.data.tags));
  const categories = await ensureTaxonomy(
    "/apis/content.halo.run/v1alpha1/categories",
    "Category",
    "category",
    categoryNames,
    existingCategories,
  );
  const tags = await ensureTaxonomy(
    "/apis/content.halo.run/v1alpha1/tags",
    "Tag",
    "tag",
    tagNames,
    existingTags,
  );

  const bySlug = new Map(existingPosts.map((post) => [post.spec.slug, post]));
  const summary = { created: 0, draft: 0, published: 0, skipped: 0, updated: 0 };

  for (const sourcePost of posts) {
    const { data, filename, frontMatter, slug } = sourcePost;
    if (!data.title) throw new Error(`${filename} has no title`);

    const raw = replaceContentLinks(sourcePost.body, imageUrls).trimStart();
    const html = await marked.parse(raw);
    const shouldPublish = data.haloPublished === true && data.draft !== true;
    const checksum = createHash("sha256")
      .update(JSON.stringify({ data, html, raw }))
      .digest("hex");
    const existing = bySlug.get(slug);
    const annotations = {
      ...(existing?.metadata.annotations || {}),
      "migration.diycv.top/source-path": `content/posts/${filename}`,
      "migration.diycv.top/source-sha256": checksum,
    };
    const currentPublished = existing?.status?.phase === "PUBLISHED";

    if (
      existing?.metadata.annotations?.["migration.diycv.top/source-sha256"] === checksum &&
      currentPublished === shouldPublish
    ) {
      summary.skipped += 1;
      continue;
    }

    const post = existing || {
      apiVersion: "content.halo.run/v1alpha1",
      kind: "Post",
      metadata: { name: stableName("github-post", slug) },
      spec: {},
    };
    post.metadata.annotations = annotations;
    post.spec = {
      ...post.spec,
      allowComment: true,
      categories: asList(data.categories).map((name) => categories.get(name)),
      cover: data.cover ? imageUrls.get(data.cover) || data.cover : "",
      deleted: false,
      excerpt: { autoGenerate: !data.description, raw: data.description || "" },
      htmlMetas: post.spec.htmlMetas || [],
      pinned: Boolean(data.pinned),
      priority: Number(data.priority || 0),
      publish: currentPublished,
      publishTime: sourcePublishTime(frontMatter),
      slug,
      tags: asList(data.tags).map((name) => tags.get(name)),
      template: post.spec.template || "",
      title: String(data.title),
      visible: "PUBLIC",
    };

    const request = {
      content: { content: html, raw, rawType: "MARKDOWN" },
      post,
    };
    let saved;
    if (existing) {
      saved = await haloRequest(`/apis/api.console.halo.run/v1alpha1/posts/${post.metadata.name}`, {
        method: "PUT",
        body: JSON.stringify(request),
      });
      summary.updated += 1;
    } else {
      saved = await haloRequest("/apis/api.console.halo.run/v1alpha1/posts", {
        method: "POST",
        body: JSON.stringify(request),
      });
      summary.created += 1;
      bySlug.set(slug, saved);
    }

    if (shouldPublish) {
      await haloRequest(`/apis/api.console.halo.run/v1alpha1/posts/${saved.metadata.name}/publish`, {
        method: "PUT",
      });
      summary.published += 1;
    } else {
      if (saved.status?.phase === "PUBLISHED") {
        await haloRequest(`/apis/api.console.halo.run/v1alpha1/posts/${saved.metadata.name}/unpublish`, {
          method: "PUT",
        });
      }
      summary.draft += 1;
    }
    process.stdout.write(`${shouldPublish ? "Published" : "Drafted"} ${slug}\n`);
  }

  process.stdout.write(`${JSON.stringify({ attachments: imageUrls.size, posts: posts.length, ...summary })}\n`);
}

await main();
