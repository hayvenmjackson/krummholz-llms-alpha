/* This worker controls only the Sebago app directory. */
const cacheName = "sebago-field-reader-790de345686569956b1f";
const files = ["./", "./manifest.webmanifest", "./icon-192.png", "./icon-512.png", "./icon-180.png"];
const urls = new Set(files.map(path => new URL(path, self.registration.scope).href));

self.addEventListener("install", event => {
  event.waitUntil((async () => {
    const cache = await caches.open(cacheName);
    await cache.addAll(files.map(path => new Request(new URL(path, self.registration.scope), {cache:"reload"})));
    await self.skipWaiting();
  })());
});

self.addEventListener("activate", event => {
  event.waitUntil((async () => {
    for (const name of await caches.keys()) {
      if (name.startsWith("sebago-field-reader-") && name !== cacheName) await caches.delete(name);
    }
    await self.clients.claim();
  })());
});

self.addEventListener("fetch", event => {
  if (event.request.method !== "GET") return;
  const url = new URL(event.request.url);
  const app = new URL("./", self.registration.scope);
  const isEntry = event.request.mode === "navigate" && url.origin === app.origin &&
                  (url.pathname === app.pathname || url.pathname === app.pathname + "index.html");
  if (!isEntry && !urls.has(url.href)) return;
  event.respondWith((async () => {
    const cache = await caches.open(cacheName);
    const result = await cache.match(isEntry ? app.href : url.href);
    return result || fetch(event.request);
  })());
});
