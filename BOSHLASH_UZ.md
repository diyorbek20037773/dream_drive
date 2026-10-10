# DreamDrive Claude Code paketini ishga tushirish

Bu paket ilova kodi emas. Bu Claude Code haqiqiy ilovani yozishi uchun tayyorlangan topshiriq, arxitektura va tekshiruv shartlari. Kodni Claude yozadi, ishga tushiradi, QA qiladi va mavjud loginlar bilan GitHub hamda Docker Hub'ga chiqaradi.

Tasdiqlangan sozlamalar:
- GitHub: https://github.com/diyorbek20037773/dream_drive
- Docker Hub: diyorbek7377
- GPU: RTX 4050, 6 GB VRAM
- Server: hali olinmagan
- CPU va yengil simulyator asosiy portativ rejim, CUDA tezlatish alohida profil

## 1 Paketni joylash
Ubuntu WSL terminalida loyiha papkasini oching. WSL fayl tizimidagi `~/projects` odatda Linux konteynerlari bilan qulayroq. D diskini ishlatmoqchi bo‘lsangiz, uning WSL yo‘li `/mnt/d/...` bo‘ladi.

Repo hali kompyuterda yo‘q bo‘lsa:
```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/diyorbek20037773/dream_drive.git
cd dream_drive
```
Repo mavjud bo‘lsa, o‘sha papkani oching; ikkinchi nusxa yaratish shart emas.
ZIP ichidagi `DreamDrive_Claude_Kit` papkasining ICHIDAGI fayllarni repo ildiziga nusxalang. `CLAUDE.md` repo ildizida bo‘lsin. `.claude`, `.env.example`, `.gitignore`, `.gitattributes` kabi nuqta bilan boshlanadigan fayllar ham ko‘chishi kerak. Mavjud `.gitignore`, CLAUDE.md yoki config fayllari bo‘lsa, ularni ko‘r-ko‘rona almashtirmang, birlashtiring. README nomi ataylab `KIT_README.md`; mavjud README o‘chmaydi.

## 2 Docker va login
Docker Desktop ishlayotgan va Ubuntu uchun WSL integration yoqilgan bo‘lsin.
```bash
docker version
docker compose version
```
GitHub login kerak bo‘lsa:
```bash
gh auth login
gh auth status
```
Docker Hub login kerak bo‘lsa:
```bash
docker login -u diyorbek7377
```
Docker so‘ragan parol/tokenni terminalning o‘zida kiriting. Uni promptga, GitHub'ga yoki menga yubormang. Docker Hub yozish huquqi uchun scoped access token ishlatilishi mumkin. `gh` yo‘q bo‘lsa, Claude o‘rnatish yo‘lini ko‘rsatadi; mavjud git credential login ham ishlashi mumkin.

## 3 Claude Code topshirig‘i
Repo ichida:
```bash
claude
```
`PROMPT_FOR_CLAUDE_CODE.md` faylining to‘liq matnini Claude Code suhbatiga joylang. CLI loyihaga ishonch yoki zarur amal ruxsatini so‘rasa, uning aniq amalini ko‘rib tasdiqlang. `--dangerously-skip-permissions` kerak emas.

## 4 Natija qanday bo‘lishi kerak
Claude `docs/IMPLEMENTATION_PLAN.md` bo‘yicha kod yozadi. U hech qanday soxta inference/FPS bermasligi, barcha tugmalarni ishlatishi, real model va yangi MP4 bilan tekshirishi, Docker'da toza ishga tushirishni isbotlashi kerak. GitHub push mahalliy QA'dan keyin, Docker Hub publish tegishli CI'dan keyin bo‘ladi.

`make qa`, Docker compose komandalarining ilova uchun Makefile/Compose fayllarini Claude YARATADI. Hozirgi paket ulardan tayyor ishlaydigan ilova borligini da’vo qilmaydi.

## 5 Server uchun keyin kerak bo‘ladigan ma’lumotlar
Server olingach: host/IP, OS, CPU/RAM/disk, domen, SSH user, lokal SSH key yo‘li, kerakli portlar. Maxfiy key/tokenlar chat orqali yuborilmaydi. Claude tayyor prod config orqali image'larni digest bilan pull qilib joylashtiradi. Oddiy CPU VPS navbatli video tahlili uchun mos bo‘lishi mumkin; real throughput benchmark bilan aniqlanadi. CARLA uchun alohida GPU infratuzilmasi kerak bo‘lishi mumkin.

Hujjatdagi 11-oktabr 13:00 muddati dastlabki ma’lumotnomadan olingan. Paketni keyinroq ishlatsangiz, Claude uni eskirgan muddat sifatida belgilashi kerak.
