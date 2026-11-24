"""Bundle the browser app and its checkpoint into a file that opens offline."""
from pathlib import Path
import base64
import hashlib
import html
import json
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"


def data(path, mime):
    return "data:" + mime + ";base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def icon(shape):
    return '<svg class="ico" aria-hidden="true" viewBox="0 0 24 24">' + shape + '</svg>'


weights = (ROOT / "sebago-40k.safetensors").read_bytes()
report = json.loads((ROOT / "evaluation.json").read_bytes())
digest = hashlib.sha256(weights).hexdigest()
assert digest == report["weights_sha256"]
values = {
    "PHOTO": data(ASSETS / "sebago-lake.webp", "image/webp"),
    "LOGO": data(ASSETS / "brand.png", "image/png"),
    "DISPLAYFONT": data(ASSETS / "librefranklin.ttf", "font/ttf"),
    "BODYFONT": data(ASSETS / "dmsans.ttf", "font/ttf"),
    "FONTLICENSE": html.escape("\n\n".join((ASSETS / f"{font}-OFL.txt").read_text("utf-8-sig") for font in ("dmsans", "librefranklin"))),
    "WEIGHTS": base64.b64encode(weights).decode("ascii"),
    "HASH": digest,
    "PRESENT": str(report["trained"]["correct_present"]),
    "ABSENT": str(report["trained"]["correct_absent"]),
    "RUNTIME": (ROOT / "model.js").read_text("utf8"),
    "UI": (ROOT / "app.js").read_text("utf8"),
    "ICON_BOOK": icon('<path d="M3 4h7a3 3 0 0 1 2 1.1A3 3 0 0 1 14 4h7v14h-7a3 3 0 0 0-2 1.1 3 3 0 0 0-2-1.1H3zM12 6v13"/>'),
    "ICON_CPU": icon('<rect x="5" y="5" width="14" height="14" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M8 2v3m4-3v3m4-3v3M8 19v3m4-3v3m4-3v3M2 8h3m-3 4h3m-3 4h3m14-8h3m-3 4h3m-3 4h3"/>'),
    "ICON_ARROW": icon('<path d="M4 12h15m-6-6 6 6-6 6"/>'),
}

# The hosted page installs as a web app. The portable file never registers a worker.
pwa = ROOT / "web"
pwa.mkdir(exist_ok=True)
worker = (ROOT / "worker.template.js").read_text("utf8")
manifest = {
    "id": "./", "name": "Sebago by Krummholz AI", "short_name": "Sebago",
    "description": "Read a field record on your own device.",
    "start_url": "./", "scope": "./", "display": "standalone",
    "background_color": "#f3f5ee", "theme_color": "#0c1411",
    "icons": [{"src": f"icon-{size}.png", "sizes": f"{size}x{size}", "type": "image/png",
               "purpose": "any maskable"} for size in (192, 512)],
}
(pwa / "manifest.webmanifest").write_text(json.dumps(manifest, indent=2) + "\n", "utf8", newline="\n")
for size in (180, 192, 512):
    shutil.copyfile(ASSETS / f"icon-{size}.png", pwa / f"icon-{size}.png")

values["ITALICFONT"] = data(ASSETS / "librefranklin-italic.ttf", "font/ttf")
values["FONTLICENSE"] = html.escape((ROOT / "NOTICE.txt").read_text("utf8") + "\n" +
                                  (ROOT / "LICENSE").read_text("utf8")) + "\n\n" + values["FONTLICENSE"]
template = (ROOT / "app.template.html").read_text("utf8")


def render(options):
    return re.sub(r"\[\[([A-Z_]+)\]\]", lambda m: options[m[1]], template).encode("utf8")


portable = render(values | {"INSTALL_HEAD": "", "INSTALL_HELP": "", "INSTALL_SCRIPT": ""})
(ROOT / "Sebago.html").write_bytes(portable)
online = render(values | {
    "INSTALL_HEAD": '<link rel="manifest" href="./manifest.webmanifest"><link rel="apple-touch-icon" href="./icon-180.png">',
    "INSTALL_HELP": (ROOT / "install.html").read_text("utf8"),
    "INSTALL_SCRIPT": """<script>
const installHelp = document.getElementById("install");
function showInstall() { if (location.hash === "#install") { installHelp.open = true; installHelp.scrollIntoView(); } }
addEventListener("hashchange", showInstall);
addEventListener("load", () => {
  showInstall();
  const status = document.getElementById("installState");
  if (!("serviceWorker" in navigator)) {
    status.textContent = "This browser cannot save the web app for offline use. The desktop download runs offline.";
    return;
  }
  navigator.serviceWorker.register("./sw.js").then(() => navigator.serviceWorker.ready).then(() => {
    status.textContent = "Offline copy ready on this browser.";
  }).catch(() => {
    status.textContent = "Offline copy could not be saved. Reopen this page online and check that browser storage is enabled.";
  });
});</script>""",
})
(pwa / "index.html").write_bytes(online)
version = hashlib.sha256(online + worker.encode() + (pwa / "manifest.webmanifest").read_bytes() +
                         b"".join((pwa / f"icon-{size}.png").read_bytes() for size in (180, 192, 512))).hexdigest()[:20]
(pwa / "sw.js").write_text(worker.replace("[[VERSION]]", version), "utf8", newline="\n")
with zipfile.ZipFile(ROOT / "Sebago-offline.zip", "w") as archive:
    for name in ("Sebago.html", "INSTALL.txt", "NOTICE.txt", "LICENSE", "assets/dmsans-OFL.txt", "assets/librefranklin-OFL.txt"):
        info = zipfile.ZipInfo(name, date_time=(2026,11,24,0,0,0))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, (ROOT / name).read_bytes())

with zipfile.ZipFile(ROOT / "Sebago-web-app.zip", "w") as archive:
    for entry in sorted(pwa.iterdir()):
        if entry.name in {"index.html", "manifest.webmanifest", "icon-180.png", "icon-192.png", "icon-512.png", "sw.js"}:
            info = zipfile.ZipInfo(entry.name, date_time=(2026,11,24,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, entry.read_bytes())
    for name in ("INSTALL.txt", "NOTICE.txt", "LICENSE"):
        info = zipfile.ZipInfo(name, date_time=(2026,11,24,0,0,0))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, (ROOT / name).read_bytes())
print(f"Built Sebago.html ({len(portable):,} bytes), both ZIP downloads and hosted app in web/.")
