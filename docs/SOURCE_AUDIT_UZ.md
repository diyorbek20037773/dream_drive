# Ma’lumotnoma tahlili va qabul qilingan qarorlar

Ma’lumotnoma loyiha maqsadini to‘g‘ri ajratadi: MP4 tahlili va interaktiv simulyatorda virtual boshqaruv. Video → perception → qaror → boshqaruv → yangi kadr zanjiri ilovaning asosiy talabidir. Undagi arxitektura va barcha metrikalar ishlab chiqish rejasi; bajarilgan natijalar sifatida olinmaydi.

## Aniqlashtirilgan joylar
1. RTX 4050 6 GB mavjud. CARLA 0.9.16 quick-start 8 GB yoki ko‘proq VRAM tavsiya qiladi; UE5 yanada og‘ir. CARLA majburiy default bo‘lsa, portativlik va vaqt talabi xavf ostida qoladi. Shuning uchun LiteRoad CPU'da ishlaydigan asosiy simulyator, CARLA esa alohida integratsiya profili.
2. YOLOPv2 demo kodi TorchScript formatidagi checkpoint yuklaydi. Oddiy Ultralytics YOLO loader yoki taxmin qilingan output shakli bilan almashtirish mumkin emas. Preprocessing, detection decoding va masklarni original kadrga qaytarish upstream bilan tekshiriladi.
3. Ma’lumotnomadagi car, bus, person va boshqa sinflar ro‘yxati model qobiliyatining isboti emas. Aniq checkpoint sinflari tekshiriladi. Person aniqlash yo‘q bo‘lsa, UI va testlar unsupported deydi; RF-DETR Nano faqat shu ehtiyoj isbotlanganda qo‘shiladi.
4. YOLOPv2 real yo‘l videosida o‘qitilgan; uning sodda simulyator RGB tasvirida ishlashi hali isbotlanmagan. LiteRoad'da rang va geometriya asosidagi haqiqiy piksel tahlili alohida `synthetic_rgb_cv` deb belgilanadi. YOLOPv2 orqali simulyatsiya boshqaruvi faqat tegishli testlardan keyin ochiladi.
5. Piksel masofa, confidence yoki nisbiy depth avtomatik ravishda metr, xavfsizlik ehtimoli yoki real tezlik emas. TTC ham yetarli geometrik/temporal shartlar bo‘lmasa `null` bo‘ladi.
6. Docker OS, dependency va image muhitini qayta yaratadi, GPU/VRAM/tezlikni barcha kompyuterlarda bir xil qilmaydi. CPU default, GPU profil va qo‘llab-quvvatlangan arxitektura alohida tekshiriladi.
7. MLOps pretrained modeldan boshlab ham zarur: model hash, dataset manifest, evaluation, MLflow va regressiya nazorati. Yangi modelni majburan o‘qitish yoki Kubernetes qo‘shish hozirgi maqsadga yordam bermaydi.
8. Server hali yo‘q. Xarid/budget/domen tanlovi ochiq; deploy fayllari va mahalliy sinovlar hozir bajariladi, tashqi serverdagi natija keyin tekshiriladi.

## Tekshiruv holati
2026-10-10 kuni GitHub web o‘qilishi muvaffaqiyatli bo‘lmadi. `git ls-remote ... HEAD` ref qaytarmadi, `git ls-remote --exit-code --heads ...` 2 kodi bilan branch refs topmadi. Bu repo boshlarining o‘sha tekshiruvdagi holati; Claude qayta audit qiladi. Kod mavjud bo‘lishi yoki repo keyin o‘zgargan bo‘lishi mumkin.

Deadline va hackathon mezonlari manba hujjatdan olingan, mustaqil tasdiqlangan yangilik emas. Lucid bilan nom bandligi haqidagi eslatma saqlanadi; bu paket savdo belgisi bo‘yicha xulosa bermaydi.

Rasmiy texnik manbalar `docs/RESEARCH_SOURCES.md` ichida. App tezligi, GPU xotirasi, model aniqligi yoki safar muvaffaqiyati hali o‘lchanmagan.
