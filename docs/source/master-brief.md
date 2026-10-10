# Owner supplied DreamDrive master brief

Extracted from the attached Word file. Claims and competition rules are source context, not verified implementation evidence.

DreamDrive AI — loyiha bo‘yicha to‘liq texnik va biznes hujjat
Quyidagi ma’lumotlar yangi ChatGPT suhbatida DreamDrive AI loyihasini boshidan tushuntirishga mo‘ljallangan. Unda hozirgacha kelishgan qarorlarimiz, modellar, arxitektura, ish bosqichlari va WIUT Hackathon talablari jamlangan.
Eng muhim kelishuvlarimiz:
Loyiha nomi: DreamDrive AI (ishchi nom).
Asosiy g‘oya: foydalanuvchi manzilni belgilaydi, AI avtomobil kamerasidan video olib, yo‘lni tahlil qiladi va virtual avtomobilni manzil tomon mustaqil boshqaradi.
Hackathon yo‘nalishi: Computer Vision — Automotive Business / Road Safety.
Asosiy input: MP4 yo‘l videosi yoki simulyatorning jonli RGB video oqimi.
Asosiy texnologiyalar: YOLOPv2, ByteTrack, OpenCV, Python, navigation/planning algoritmlari va CARLA.
Maqsad: faqat video tahlil emas, balki Video → Perception → Decision → Action → Feedback zanjiri.
Chegara: bu haqiqiy avtomobilga o‘rnatiladigan, haydovchi uxlayotganida harakatlana olishi tasdiqlangan avtopilot emas. Hackathon uchun simulyatsion prototip.
Holat: quyida bayon etilgan arxitektura — ishlab chiqish rejasi. Modellar hali shu suhbatda ishga tushirilmagan, integratsiya va natijalar tekshirilmagan. Yangi chat ham ushbu farqni saqlashi kerak.
Texnik manbalarni ham tekshirdim: YOLOPv2 bitta modelda obyekt, yo‘l maydoni va chiziqlarini aniqlashga mo‘ljallangan; CARLA kamera va avtomobilni boshqarish interfeyslarini beradi; RF-DETR Nano esa kerak bo‘lsa qo‘shimcha object detection modeli bo‘ladi. 

GitHub
+3
Quyidagi hujjatni to‘liq nusxalab, yangi chatga yuborishingiz mumkin.
DREAMDRIVE AI
Complete Project Master Brief — WIUT Hackathon 2026
Project name: DreamDrive AI (ishchi nom)
Full title: DreamDrive AI — Video-Based Autonomous Navigation & Driving Agent
Category: Computer Vision / Autonomous Driving / Agentic AI / Road Safety
Participation mode: Startup
Project stage: Idea selected — architecture and implementation planning
Hackathon deadline: 11 October 2026, 13:00, Asia/Tashkent
Important: The architecture described below is planned, not yet implemented or validated.

1. LOYIHANING ASOSIY G‘OYASI
DreamDrive AI — avtomobil kamerasidan olingan video tasvirlarni Computer Vision orqali qayta ishlab, yo‘l holatini tushunadigan, navigatsiya bo‘yicha qarorlar qabul qiladigan va virtual avtomobilni belgilangan manzilga mustaqil boshqaradigan AI agent tizimi.
1.1. Muammo
An’anaviy avtomobillar haydovchining doimiy boshqaruvini talab qiladi. Uzoq safar, charchoq, murakkab yo‘l sharoiti va diqqatning pasayishi haydash xavfsizligiga ta’sir qiladi.
Hozir ayrim avtomobillarda adaptiv kruiz nazorati, qatorni ushlab turish va navigatsiya yordamidagi haydash funksiyalari mavjud. Ammo bularning hammasi haydovchi uxlay oladigan to‘liq avtonom transport emas.
1.2. Uzoq muddatli maqsad
Foydalanuvchi navigatorda manzil tanlaydi. Avtomobil atrofdagi vaziyatni mustaqil tushunadi, marshrutni rejalashtiradi va tegishli xavfsizlik talablari bajarilgan sharoitdagina inson boshqaruvisiz manzilga yetib boradi.
Bu kelajakdagi Level 4 avtonom boshqaruvga oid tadqiqot yo‘nalishi. Level 4 faqat tasdiqlangan operatsion sharoitlarda haydovchisiz ishlashi mumkin.
Biz hackathon davomida bu darajadagi real avtomobilni yaratmaymiz.
1.3. Hackathon uchun maqsad
DreamDrive AI Autonomous Driving Simulation Prototype yaratish.
Tizim quyidagilarni ko‘rsatishi kerak:
Foydalanuvchi virtual xaritada manzil belgilaydi.
AI old kameradan video oladi.
Computer Vision yo‘l chiziqlari, avtomobillar va boshqa obyektlarni aniqlaydi.
Navigation agent marshrutni hisoblaydi.
Decision agent keyingi harakatni tanlaydi.
Safety supervisor qarorni tekshiradi.
Controller virtual avtomobilni boshqaradi.
Virtual avtomobilning yangi holati yana kuzatiladi.
Safar muvaffaqiyati va xavfsizlik ko‘rsatkichlari o‘lchanadi.
Loyihaning farqi: oddiy video annotation emas, perception-to-action closed-loop simulation.

2. HACKATHON SHARTLARI
WIUT Hackathon 2026 finalida biz Computer Vision track bo‘yicha qatnashmoqdamiz.
Hamkorlar: Toyota Royal Motors, BYD Royal Motors va Ducati.
Topshiriqning mohiyati:
Build a product that turns images or video into a decision or an action for the automotive business or for road safety.
DreamDrive AI ushbu talabga mos keladi, chunki video asosiy ma’lumot manbai va undan transportni boshqarishga oid virtual harakat qarorlari chiqariladi.
2.1. Baholash mezonlari
Mezon
Ulushi
Technical complexity
25%
Innovation and originality
25%
Impact and practicality
25%
Presentation
15%
Business potential
10%
Texnologiya va innovatsiya jami 50%.
Shuning uchun oddiy tayyor YOLO modelini web interfeysga ulash bilan cheklanish mumkin emas.
2.2. Majburiy topshiriladigan narsalar
Public HTTPS orqali ochiladigan, haqiqiy inference bajaradigan prototype.
PDF yoki PPTX formatidagi defence presentation.
Public GitHub repository: frontend, backend, README, ma’lumot manbalari va litsenziyalar.
Hakam o‘zining yangi video inputi bilan tizimni tekshira olishi kerak.
Himoya: 7 daqiqa, savol-javob: 3 daqiqa.
Deadline: 11 October 2026, 13:00 Tashkent time.
Belgilangan muddatdan keyin submitted commitni o‘zgartirib, yangi kod bilan deploy qilish mumkin emas.
2.3. Cheklovlar
Quyidagilar mumkin emas:
Hard-coded AI natijalari.
Haqiqiy deb ko‘rsatilgan ishlamaydigan demo.
O‘ylab topilgan aniqlik, FPS yoki biznes metrikalari.
Rozilik olinmagan shaxsiy tasvirlar yoki avtomobil davlat raqamlari.
Eski ishlarni final davrida yaratilgandek ko‘rsatish.
Ochiq modellar va kodlardan manbasini ko‘rsatmasdan foydalanish.
Oldindan yaratilgan ochiq modellar va kutubxonalardan foydalanish mumkin. Ularning kelib chiqishi va litsenziyasi README’da ko‘rsatiladi.

3. DREAMDRIVE'NING IKKITA REJIMI
Bitta muhim texnik haqiqat bor:
Oddiy MP4 video orqali o'tmishda tasvirga olingan avtomobilni boshqarish mumkin emas.
Avtomobilga berilgan yangi buyruq keyingi video kadrga ta’sir qilishi uchun jonli interaktiv muhit zarur.
Shuning uchun DreamDrive ikki rejimdan tashkil topadi.
MODE A — VIDEO ANALYSIS
Input: foydalanuvchi yuklagan dashcam videosi.
Masalan:
road_video.mp4
Tizim:
Videoni qabul qiladi.
Kadrlarga ajratadi.
Obyektlarni aniqlaydi.
Yo‘l chiziqlarini topadi.
Obyektlarni kuzatadi.
Yo‘l sahnasining strukturali ifodasini yaratadi.
Ehtimoliy xavfli holatlarni ajratadi.
Harakatga oid maslahat yoki tahliliy qaror chiqaradi.
Annotatsiyalangan video va hisobot yaratadi.
Output:
Annotated video.
Vehicle/person detection.
Lane/drivable-area overlay.
Frame timestamps.
Decision timeline.
Confidence va uncertainty.
Driving analysis report.
Bu rejim haqiqiy avtomobilni boshqarmaydi. Unda qarorlar tahliliy tavsiya hisoblanadi.
MODE B — AUTONOMOUS SIMULATION
Input:
Simulyatorning jonli RGB kamera oqimi.
Foydalanuvchi belgilagan virtual manzil.
Simulyatorning kerakli lokalizatsiya va holat telemetriyasi.
Tizim:
Boshlang‘ich nuqtadan manzilgacha yo‘l hisoblaydi.
Jonli RGB kameradan kadr oladi.
Vision engine orqali yo‘l va obyektlarni aniqlaydi.
World state hosil qiladi.
Qaysi harakatni bajarishni rejalashtiradi.
Safety supervisor buyruqni tekshiradi.
Virtual rul, gaz va tormozga buyruq uzatadi.
Yangi kamera kadrini oladi.
Maqsadga yetguncha jarayonni takrorlaydi yoki xavfsiz holatga o‘tadi.
Muhim: simulyator telemetriyasi virtual avtomobil joylashuvini aniqlashda ishlatilishi mumkin, lekin tayyor simulyator obyektlari koordinatalari Computer Vision orqali aniqlangandek ko‘rsatilmaydi.

4. TO‘LIQ ARXITEKTURA
DreamDrive quyidagi asosiy qismlardan iborat.
INPUT LAYER
MP4 upload / Live RGB camera / Destination / Simulator localization.
↓
VIDEO PROCESSING LAYER
OpenCV + FFmpeg
↓
COMPUTER VISION LAYER
YOLOPv2 + optional RF-DETR + ByteTrack
↓
WORLD MODEL LAYER
Scene fusion + temporal tracking + uncertainty estimation
↓
DECISION & NAVIGATION LAYER
A* / Waypoint planning / Finite-State Machine
↓
SAFETY SUPERVISOR
Deterministic rules + fallback state
↓
DRIVING CONTROLLER
Pure Pursuit / PID
↓
CARLA SIMULATOR
Virtual steering / throttle / brake / vehicle motion
↓
FEEDBACK LOOP
New frames → New perception → New decision
Qwen3-VL alohida explanation/reporting agent sifatida ishlatilishi mumkin, ammo real-time steering controllerga bevosita ulanmaydi.

5. HAR BIR BOSQICH VA TEXNOLOGIYA
STEP 1 — Video Ingestion
Technology: OpenCV + FFmpeg
Vazifasi
Video faylni qabul qilish yoki jonli RGB kameradan kadr olish.
Nega OpenCV?
Python bilan ishlaydi.
Kadrlarga ajratadi.
Resize va preprocessing qiladi.
Bounding box va maskalarni chizadi.
Video annotation uchun qulay.
Model inference pipeline bilan oson integratsiyalashadi.
Nega FFmpeg?
Turli video formatlari.
Video decoding/encoding.
FPS va resolution o‘zgartirish.
Annotatsiyalangan MP4 eksporti.
Input
MP4 / MOV / RGB stream
Output
Frame + timestamp + frame quality
Bu bosqichda AI modeli majburiy emas.

STEP 2 — Object Detection
Primary model: YOLOPv2
Optional model: RF-DETR Nano
Vazifasi
Video ichidagi ko‘rinadigan transport vositalari va boshqa o‘qitilgan toifadagi obyektlarni aniqlash.
Misollar:
Car.
Bus.
Truck.
Motorcycle.
Person.
Bicycle.
YOLOPv2 nima uchun?
YOLOPv2 bir vaqtning o‘zida quyidagi vazifalarga mo‘ljallangan:
Traffic object detection.
Drivable-area segmentation.
Lane detection.
Shuning uchun birinchi MVP uchun yagona ko‘rish modelidan boshlash mumkin.
Repository:https://github.com/CAIC-AD/YOLOPv2
RF-DETR Nano nima uchun?
Agar YOLOPv2 ma’lum toifadagi obyektlarni yetarlicha aniqlamasa, RF-DETR Nano qo‘shimcha object detector sifatida solishtirib ko‘riladi.
Afzalliklari:
Pretrained detection modeli.
Transformer arxitekturasi.
Fine-tuning imkoniyati.
Core Nano modeli Apache 2.0 litsenziyali.
Repository:https://github.com/roboflow/rf-detr
Muhim: ikkala modelni bir vaqtning o‘zida ishlatish majburiy emas. Avval birini benchmark qilib, qo‘shimcha model kiritish natijani yaxshilashini isbotlash kerak.
Output
class_id, bounding_box, confidence

STEP 3 — Lane Detection & Drivable Area
Model: YOLOPv2
Vazifasi
AI quyidagi elementlarni aniqlaydi:
Yo‘l chiziqlari.
Ko‘rinadigan harakatlanish maydoni.
Avtomobil qatorining taxminiy chegaralari.
Nega?
Avtomobil oldidagi obyektlarni topishning o‘zi yetmaydi. Virtual controller qaysi hudud yo‘l ekanini va qator qanday joylashganini bilishi kerak.
Output
lane_mask + drivable_area_mask
Cheklov
Ko‘rinadigan asfalt maydoni avtomatik ravishda qonuniy va xavfsiz harakat hududi degani emas.
Model tungi video, yomg‘ir, yo‘l chiziqlari o‘chgan hudud va CARLA grafikasi uchun alohida sinovdan o‘tishi kerak.

STEP 4 — Object Tracking
Algorithm: ByteTrack
Repository:https://github.com/FoundationVision/ByteTrack
Vazifasi
Bir avtomobil turli kadrlarda qayta aniqlansa, ularni bitta obyekt sifatida kuzatish.
Masalan:
Frame 1: Car #5
Frame 2: Car #5
Frame 3: Car #5
Bu orqali obyekt tasvir bo‘ylab qayerga harakatlanayotganini hisoblash mumkin.
Nega ByteTrack?
Tayyor object detection natijalari bilan ishlaydi.
Multi-object tracking uchun yaratilgan.
Tracking ID beradi.
Vaqt bo‘yicha kuzatuv imkonini yaratadi.
Output
track_id + position_history + motion_in_image
Muhim: piksel bo‘yicha harakat haqiqiy metr/soniya tezlikka teng emas.

STEP 5 — Depth Estimation
Optional model: Depth Anything V2 Small
Repository:https://github.com/DepthAnything/Depth-Anything-V2
Vazifasi
Bitta RGB kadri orqali sahnadagi yaqin va uzoq hududlarning nisbiy chuqurlik xaritasini yaratish.
Nega Small?
Nisbatan ixcham.
Qo‘shimcha perception tajribalari uchun mos.
Small vaznlari permissive Apache 2.0 litsenziyali.
Cheklov
Oddiy Depth Anything V2 nisbiy chuqurlik beradi. Uni real masofa deb talqin qilib, avtomobil tormozini boshqarish mumkin emas.
MVP'da bu modulni majburiy qilmaymiz.
Simulyatsiyada real masofani baholashni tekshirish uchun alohida ground-truth sensorlardan foydalanish mumkin. Ground truth model inputi va evaluation ma’lumotlari sifatida aniq ajratiladi.

STEP 6 — World State Builder
Technology: Python + NumPy + data fusion logic
Bu alohida tayyor LLM emas, biz yozadigan strukturali holat modeli.
Vazifasi
Barcha perception natijalarini bitta izchil ko‘rinishga keltirish.
Masalan:
Oldinda transport mavjudligi.
Yo‘l chizig‘i aniqlanganligi.
Video sifati.
Obyektlarning track ID'lari.
Qaysi hudud harakatlanish maydoni ekani.
Aniqlik va noaniqlik signallari.
Oldingi kadrlardagi kuzatuvlar.
Nega kerak?
YOLOPv2 faqat joriy kadr haqida ma’lumot beradi.
Avtonom tizimga esa oldingi holatlar ham kerak.
World State Builder vaqt bo‘yicha filtering va observation fusion vazifasini bajaradi.
Output
StructuredWorldState
Bu ma’lumot Decision Agent uchun asos bo‘ladi.

STEP 7 — Vision-Language Reasoning
Optional model: Qwen3-VL-4B-Instruct
Model:https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
Vazifasi
Murakkab video epizodlarini foydalanuvchi tushunadigan tilda sharhlash.
Masalan:
"Old kamera tasvirida avtomobil va yo‘l chiziqlari aniqlangan. Bir necha kadrda qator chizig‘i yaxshi ko‘rinmayapti. Shu sababli epizod qo‘shimcha tekshiruvga yuborildi."
Nega Qwen3-VL?
Vision-language model.
Rasm va video ma’lumotlari bilan ishlaydi.
Hodisalarni izohlash mumkin.
JSON yoki strukturali hisobot shakllantirishda ishlatilishi mumkin.
Lokal GPU yoki serverda o‘rnatish imkoniyati mavjud.
Nima uchun driving loopga bevosita qo‘shilmaydi?
LLM:
noto‘g‘ri taxmin qilishi;
javobni kechiktirishi;
bir xil vaziyatga turli javob berishi;
xavfsizlik talablarini kafolatlamasligi mumkin.
Shu sabab Qwen3-VL asosan:
Event analysis.
Explanation.
Operator assistant.
Post-trip report.
uchun ishlatiladi.
MVP'da ushbu modul ixtiyoriy.

STEP 8 — Navigation Agent
Algorithm: A yoki Dijkstra*
Vazifasi
Virtual xaritada foydalanuvchi tanlagan start va destination nuqtalari orasidagi marshrutni hisoblash.
Misol:
A → B → C → D
Nega A*?
Graph-search algoritmi.
Maqsadga yo‘naltirilgan.
Nazorat va test qilish oson.
LLMga qaraganda deterministik.
Input
Start location.
Destination.
Virtual road graph.
Output
Route waypoints
CARLA yo‘l xaritasi va lokal marshrut rejalashtirish vositalaridan foydalanish mumkin.

STEP 9 — Driving Decision Agent
Primary approach: Finite-State Machine (FSM)
Vazifasi
World State va Navigation Agent ma’lumotlari asosida virtual avtomobilning navbatdagi rejimini belgilash.
Rejimlar
FOLLOW_ROUTE
FOLLOW_VEHICLE
SLOW_DOWN
STOP
TURN
MINIMAL_RISK
ARRIVED
Nega FSM?
Oddiy va tushunarli.
Deterministik.
Har bir qarorni logga yozish mumkin.
Unit test qilish mumkin.
Xavfsizlik qoidalari bilan integratsiya oson.
Misol:
Agar qator aniqlanmasa va tizim ishonchi yetarli bo‘lmasa, odatdagi marshrutni davom ettirish emas, minimal-risk rejimi ko‘rib chiqiladi.
Qoidalar aniq xavfsizlik natijasini kafolatlamaydi; ularning to‘g‘ri ishlashi sinovlar bilan baholanishi kerak.

STEP 10 — Motion Planning
Technology: Waypoint Planner
Vazifasi
Navbatdagi manzil nuqtasiga harakatlanish uchun virtual trajektoriyani tanlash.
Nega kerak?
Navigation Agent butun yo‘nalishni biladi.
Motion Planner esa hozirgi holatdan navbatdagi qisqa harakatni rejalashtiradi.
MVP'da:
Waypoint following.
Basic trajectory planning.
Route deviation monitoring.
yetarli.
Murakkab Model Predictive Control yoki Reinforcement Learning dastlab majburiy emas.

STEP 11 — Driving Controller
Algorithms: Pure Pursuit + PID
Vazifasi
Virtual avtomobilga boshqaruv buyruqlarini hosil qilish.
Pure Pursuit
Virtual avtomobilni waypointlar bo‘ylab boshqarishga mo‘ljallangan lateral tracking algoritmi.
PID
Belgilangan virtual tezlikni ushlab turish uchun longitudinal control.
Output
Steering.
Throttle.
Brake.
Nega neural controller emas?
Chunki noldan neural driving policy yaratish katta dataset va juda ko‘p simulyatsion testlarni talab qiladi.
Klassik controller hackathon uchun tekshiriladigan baseline beradi.
Bu qism faqat virtual avtomobil uchun.

STEP 12 — Independent Safety Supervisor
Technology: Rule-based verification
Vazifasi
Controllerga beriladigan buyruqlarni alohida tekshirish.
Tekshiriladigan holatlar:
Kamera ma’lumotlari yo‘qolishi.
Detection ishonchliligining keskin pasayishi.
Trajektoriyadan chiqish.
World State va Navigation orasidagi nomuvofiqlik.
Tasdiqlangan virtual xavf hodisasi.
Controller xatosi.
Natija
Buyruqni ruxsat etilgan chegarada bajarish.
Harakatni cheklash.
Minimal-risk holatiga o‘tish.
Simulyatsiyani xavfsiz yakunlash yoki testni failed sifatida qayd etish.
Ushbu modulning nomi Safety Supervisor bo‘lishi real avtomobil xavfsizligini sertifikatlash degani emas.

STEP 13 — Simulation Environment
Primary simulator: CARLA
Website:https://carla.org/
Repository:https://github.com/carla-simulator/carla
Nega CARLA?
U avtonom avtomobil texnologiyalarini o‘rganish uchun mo‘ljallangan.
Asosiy imkoniyatlar:
3D yo‘l muhiti.
Virtual avtomobillar.
RGB kamera.
Virtual yo‘l tarmog‘i.
Harakat va fizika.
Steering/throttle/brake interfeysi.
Collision va lane invasion testlari.
Turli transport ssenariylari.
Versiyaga qarab ob-havo konfiguratsiyalari.
Muhim apparat talabi
CARLA'ning yangi UE5 versiyalari kuchli GPU talab qiladi. Rasmiy quick-start tavsiyasida RTX 3000+ sinfi, kamida 16 GB VRAM va taxminan 130 GB disk ko‘rsatilgan.
Bu barcha eski CARLA versiyalarining qat’iy minimal talabi emas.
Fallback
Agar CARLA o‘rnatish yoki GPU muammosi bo‘lsa:
HighwayEnv.
Yengil 2D road simulator.
Brauzerda ishlaydigan virtual avtomobil.
RGB-frame asosidagi soddalashtirilgan closed loop.
Fallback'da ham Computer Vision haqiqiy render qilingan kadrlarni qayta ishlashi kerak.
Tayyor simulyator obyekt koordinatalaridan foydalanib, buni perception natijasi deb ko‘rsatish mumkin emas.

6. ENG MUHIM TEXNIK FARQ
DreamDrive ichida uchta narsa farqlanishi shart:
A. Real AI Perception
Model kamera pikselidan obyektlarni va yo‘l strukturasini aniqlaydi.
B. Simulator Telemetry
Virtual avtomobilning pozitsiyasi, tezligi va boshqaruv holati kabi ma’lumotlar.
Bular navigatsiya va controller uchun kerak bo‘lishi mumkin.
C. Evaluation Ground Truth
Simulyator biladigan aniq obyekt joylashuvi, to‘qnashuv, yo‘l chizig‘idan chiqish kabi holatlar.
Bular AI sifati va xavfsizlik natijalarini baholash uchun kerak.
Perception natijasini ground truth bilan almashtirib, model muvaffaqiyati sifatida ko‘rsatish taqiqlanadi.

7. LIVE DEMO SSENARIYSI
7.1. Video Analysis Demo
Hakam platformaga kiradi.
Yangi MP4 video yuklaydi.
Start Analysis tugmasini bosadi.
Backend video kadrlarini qayta ishlaydi.
Yo‘l va obyektlar aniqlanadi.
Natija video ustida ko‘rsatiladi.
Decision timeline shakllanadi.
Report tayyorlanadi.
Bu yerdagi detection va qarorlar haqiqiy inference orqali hosil bo‘lishi kerak.
7.2. Autonomous Simulation Demo
Hakam virtual xaritadan destination tanlaydi.
Start Simulation tugmasini bosadi.
Virtual avtomobil harakatni boshlaydi.
Kamera feed frontendga uzatiladi.
CV inference ishlaydi.
Qarorlar panelda ko‘rsatiladi.
Controller virtual avtomobilga buyruq beradi.
Avtomobil marshrutni davom ettiradi.
Simulyatsiyada yangi avtomobil yoki to‘siq paydo bo‘lsa, tizim uni kuzatadi.
Safar muvaffaqiyatli yakunlanadi yoki failure/safety state qayd etiladi.
Demo natijasi oldindan kafolatlanmaydi.
Hakam ko‘rishi kerak bo‘lgan komponentlar
Camera feed.
Detection bounding boxes.
Lane/drivable-area mask.
Current driving state.
Planned virtual route.
Vehicle simulation status.
Safety warnings.
Measured metrics.

8. FRONTEND ARXITEKTURA
Recommended: React + TypeScript + Tailwind
Asosiy sahifalar:
Dashboard
DreamDrive umumiy ko‘rinishi.
Video Analysis
Video yuklash va inference natijalarini ko‘rish.
Autonomous Simulation
Virtual xarita, start/destination va live simulation.
AI Decisions
Agentning vaqt bo‘yicha qarorlari, sabablari va confidence ma’lumotlari.
Evaluation Dashboard
Detection sifati, latency, marshrut va collision ko‘rsatkichlari.
Trip Report
Virtual safar yakuniy hisoboti.
Dizayn yo‘nalishi
Professional automotive technology uslubi.
Toza va zamonaviy dashboard.
Kamera videosi asosiy markaziy element.
Minimalist grafikalar.
Real metrikalar.
Desktop va mobile responsive.
Hech qanday ishlamaydigan tugma yoki fake real-time metric bo‘lmasligi kerak.

9. BACKEND ARXITEKTURA
Recommended: Python + FastAPI
Asosiy modullar:
Video upload API.
Video processing worker.
Object detection service.
Lane detection service.
Tracking manager.
World-state builder.
Decision engine.
Navigation engine.
Simulator adapter.
Safety supervisor.
Evaluation engine.
Reporting service.
API konsepti
POST /api/video/analyze
Yangi videoni tahlilga qabul qiladi.
GET /api/video/{job_id}
Tahlil holatini qaytaradi.
GET /api/video/{job_id}/results
Detection, decisions va report natijalari.
POST /api/simulation/start
Virtual simulyatsiyani boshlaydi.
POST /api/simulation/destination
Virtual manzilni belgilaydi.
POST /api/simulation/stop
Simulyatsiyani to‘xtatadi.
GET /api/simulation/state
Joriy virtual avtomobil holati.
GET /api/evaluation/summary
Real test natijalari.
Jonli dashboard uchun WebSocket ishlatish mumkin.

10. REPOSITORY STRUKTURASI
Tavsiya etiladigan struktura:
dreamdrive-ai/
  frontend/
    src/
      components/
      pages/
      services/
      hooks/
      types/

  backend/
    app/
      main.py
      api/
        video.py
        simulation.py
        evaluation.py

      perception/
        video_processor.py
        yolopv2_detector.py
        object_detector.py
        tracker.py
        world_state.py

      agents/
        decision_agent.py
        navigation_agent.py
        safety_supervisor.py
        explanation_agent.py

      control/
        waypoint_planner.py
        driving_controller.py

      simulator/
        carla_adapter.py
        fallback_adapter.py

      evaluation/
        benchmark.py
        metrics.py
        scenarios.py

  tests/
    unit/
    integration/
    evaluation/

  datasets/
    README.md

  scripts/
  configs/

  README.md
  DATA_SOURCES.md
  LICENSES.md
  MODEL_CARD.md
  requirements.txt
  Dockerfile
  docker-compose.yml
Kod modulli bo‘lishi kerak.
Model yoki simulyator almashtirilsa, butun loyiha arxitekturasi qayta yozilmasligi maqsadga muvofiq.

11. DATASET VA TRAINING
Birinchi bosqichda mavjud pretrained vaznlardan foydalaniladi.
Ma’lumot manbalari
BDD100K
Avtonom haydash va yo‘lni tushunish uchun ishlatiladigan dataset. YOLOPv2 pretrained vaznlari shu yo‘nalishdagi ma’lumotlar bilan ishlab chiqilgan.
Dataset va pretrained weights litsenziyalarini alohida ko‘rib chiqish kerak.
CARLA-generated data
Simulyator orqali virtual yo‘l, mashina va to‘siq ssenariylarini yaratish.
Synthetic data bo‘lsa, README’da generation process yoziladi.
Custom evaluation videos
Tegishli ruxsat bilan olingan mustaqil dashcam videolar.
Shaxslar yuzlari va avtomobil davlat raqamlari bilan bog‘liq maxfiylik talablariga amal qilinadi.
Training strategiyasi
Pretrained baseline inference.
O‘z test videolarimizda sinash.
Failure cases aniqlash.
Zarurat bo‘lsa kichik fine-tuning.
Training va evaluation ma’lumotlarini ajratish.
Modelning eski va yangi variantlarini taqqoslash.
Bir kunlik hackathonda yangi avtopilot foundation modelini noldan o‘qitish maqsad qilinmaydi.

12. MODEL EVALUATION
Hakamlar uchun natijalar alohida ko‘rsatilishi kerak.
Komponent
Metrika
Object detection
Precision, Recall, mAP
Lane / drivable segmentation
IoU
Tracking
IDF1, ID switches
Model performance
FPS, latency
Navigation
Route completion rate
Virtual driving
Collision count
Lane control
Lane deviation / invasions
Safety supervisor
Scenario-based pass/fail
Overall system
End-to-end latency
Deployment
Request cost, GPU/CPU usage
Test sharoitlari
Daytime.
Nighttime.
Rain.
Clear road.
Vehicle ahead.
Pedestrian present.
Obstacle present.
Low-confidence scene.
Unseen user-uploaded video.
Barcha natijalar o‘zimiz yaratgan test set asosida hisoblanadi.
Uskunamizda inference tezligi qanday bo‘lsa, shunday ko‘rsatiladi.
Agar model fail bo‘lsa, yashirilmaydi.
Ablation tests
Innovatsiya va texnik murakkablikni isbotlash uchun:
Detector only.
Detector + tracking.
Detector + tracking + state-based decisions.
Full pipeline with safety supervisor.
Variantlar orasidagi farq o‘lchanadi.

13. N8N KERAKMI?
MVP'ning asosiy driving loopi uchun yo‘q.
n8n — workflow automation platformasi.
U real-time steering uchun emas, qo‘shimcha biznes avtomatlashtirish uchun qulay.
Keyinchalik:
Simulyatsiya yakunlanganda hisobot yaratish.
Xavfli hodisalar haqida operatorga xabar berish.
Test natijalarini saqlash.
Email yoki Telegram orqali bildirishnoma yuborish.
uchun ishlatish mumkin.
Asosiy driving loop:
Python + CV + decision logic + controller + simulator.
n8n faqat tashqi workflow qatlami.

14. DEPLOYMENT
Final talabi bo‘yicha public HTTPS sayt kerak.
Variant A — GPU cloud
Frontend: Vercel yoki boshqa static hosting.
Backend: GPU server.
AI inference: serverda.
Simulation: GPU serverda, resurs yetarli bo‘lsa.
Dashboard: HTTPS/WebSocket.
Variant B — Cheklangan resursdagi MVP
Frontend: public hosting.
Video inference: yengil model ishlaydigan backend.
Simulation: yengil 2D muhit.
Rendering: browser yoki server.
Decision logic: Python backend.
Muhim
CARLA'ning og‘irligi sabab uni ommaviy foydalanishga mo‘ljallangan web deploymentda ishga tushirish alohida infratuzilma talab qiladi.
Shuning uchun avval qaysi simulator va hardware bilan ishlashimizni tekshirish zarur.
Frontend va backendning submitted repository bilan bog‘liqligi aniq bo‘lishi kerak.

15. BUSINESS MODEL
Hackathon uchun dastlabki ishtirok rejimi:
STARTUP
DreamDrive AI'ni hozircha real avtomobillarga to‘liq avtopilot o‘rnatadigan tayyor tijorat mahsuloti deb sotmaymiz.
Dastlab quyidagi sohalarda pilot sinovlar o‘tkazish mumkin:
Driving simulator developers.
Automotive research and development teams.
Driving schools va professional training centers.
Universities va autonomous driving laboratories.
ADAS prototiplarini sinash tashkilotlari.
Dastlabki biznes taklifi
Autonomous Driving Simulation & Safety Evaluation Platform
Mijoz ssenariylarni virtual sharoitda sinaydi, perception natijalarini ko‘radi, driving agent qarorlarini baholaydi va test hisobotlarini oladi.
Revenue models
B2B SaaS.
Pilot integration fee.
Per simulation hour.
Enterprise licensing.
Custom simulation scenarios.
Bu faqat biznes gipotezalari.
ROI
Hisoblash uchun quyidagi haqiqiy ma’lumotlarni yig‘ish kerak:
Bir virtual testni bajarish qiymati.
GPU/server xarajati.
Muhandis mehnati.
Testni qo‘lda tayyorlash va tahlil qilish vaqti.
Avtomatlashtirish natijasida tejalgan vaqt.
Pilotning boshlang‘ich xarajati.
Formula:
Net Benefit = Verified Savings - Total Operating Costs
ROI = Net Benefit / Total Investment × 100%
Barcha farazlar alohida ko‘rsatiladi. Hozircha real ROI o‘lchanmagan.

16. RAQOBAT VA INNOVATSIYA
Avtonom haydash yangi soha emas.
Tesla, Waymo, BYD va boshqa kompaniyalar avtomatlashtirilgan haydash texnologiyalarini ishlab chiqmoqda.
Shuning uchun DreamDrive'ning innovatsiyasini "dunyoda birinchi avtopilot" deb taqdim etish noto‘g‘ri.
Bizning potensial farqimiz
Video-first perception pipeline.
Modulli navigation va safety architecture.
Auditoriya ko‘ra oladigan explainable decision log.
Camera perception va simulyator ground truth orasidagi aniq ajratish.
Failure-aware virtual driving agent.
Ochiq va qayta ishga tushiriladigan testing framework.
Lokal sharoitlar uchun keyinchalik moslashuv.
Bular hali isbotlangan ustunlik emas, mahsulotda ishlab chiqilishi va benchmark bilan tekshirilishi kerak bo‘lgan imkoniyatlar.

17. HACKATHON UCHUN MVP PRIORITETLARI
P0 — Eng muhim
Video upload.
Real inference.
Object detection.
Lane/drivable-area detection.
Decision timeline.
Annotated video.
Public web demo.
README va source attribution.
P1 — Asosiy mahsulot g‘oyasi
Destination selection.
Live RGB-based simulation.
Navigation.
Virtual driving controller.
Safety supervisor.
Collision / route test metrics.
P2 — Vaqt qolsa
Qwen3-VL explanations.
Depth Anything.
Advanced weather scenarios.
Voice assistant.
n8n notifications.
Multi-route comparison.
Agar P1 haqiqiy ishlamasa, P0'ni tayyor mahsulot sifatida ko‘rsatish mumkin, lekin uni to‘liq Autonomous Driving Agent deb noto‘g‘ri nomlash mumkin emas.

18. IMPLEMENTATION ROADMAP
Phase 1 — Environment Audit
OS, GPU, VRAM va RAM ni aniqlash.
Disk joyini tekshirish.
Python muhitini sozlash.
CARLA ishlashi mumkinligini baholash.
Simulator fallback variantini tanlash.
Phase 2 — Computer Vision
YOLOPv2 checkpointni yuklash.
Bitta rasmda inference.
MP4 videoda inference.
Lane va drivable-area natijalarini chiqarish.
FPS/latency o‘lchash.
Phase 3 — Tracking and Decisions
ByteTrack.
World State.
FSM.
Video decision timeline.
Failure handling.
Phase 4 — Simulation
Virtual avtomobil.
Route waypoints.
Controller.
RGB camera feedback.
Safety checks.
Phase 5 — Full Integration
Frontend.
Backend.
Simulation stream.
Evaluation.
Exportable results.
Phase 6 — Final Testing
Unseen video.
Scenario tests.
Night/rain limitations.
Deployment check.
Repository freeze.
Presentation.
Muhim: vaqt qisqa bo‘lgani uchun birinchi navbatda eng xavfli integratsiyalar ishlashini tekshirish kerak.

19. HAKAMLAR UCHUN PRESENTATION
7 daqiqalik himoya rejasi:
Slide 1 — Startup: DreamDrive AI
Maqsad, mahsulot va uning cheklangan simulyatsion tabiati.
Slide 2 — The Problem
Haydash xavfsizligi, avtomatlashtirilgan perception va test platformalariga ehtiyoj.
Slide 3 — Live Demo
Real video inference va imkon bo‘lsa virtual driving.
Slide 4 — Technical Architecture
Video → YOLOPv2 → World State → FSM → Planner → Safety → CARLA.
Slide 5 — Evaluation & Innovation
O‘zimizning testlarimiz, latency, route completion, xatolar va ablation.
Slide 6 — Business Model & ROI
Target customers, pilot cost, unit economics va taxminlar.
Slide 7 — Roadmap
3 oylik software-only pilot, keyingi integratsiya va tadqiqotlar.

20. ENG MUHIM XAVFLAR
20.1. Real-time inference sekinligi
Bir nechta modelni bir vaqtda ishga tushirish latency ni oshirishi mumkin.
Yechim: birinchi MVP'da minimal model stack.
20.2. Domain shift
Haqiqiy yo‘l videosida ishlagan model CARLA grafikasi bilan bir xil natija bermasligi mumkin.
Yechim: simulator-specific evaluation va zarurat bo‘lsa fine-tuning.
20.3. Camera-only uncertainty
Bitta RGB kamera orqali barcha real masofa, tezlik va xavflarni ishonchli aniqlash mumkin emas.
Yechim: simulyatsiya chegarasini aniq belgilash, noaniqliklarni qayd etish va ground truthni alohida baholashda ishlatish.
20.4. Safety
Real Level 4 avtonom boshqaruv uchun hozirgi MVP yetarli emas.
Yechim: real avtomobil actuatorslariga hech qanday buyruq yubormaslik.
20.5. Fake autonomy
CARLA'dagi tayyor autopilot yoki scripted animation'dan foydalanib, uni o‘zimizning AI boshqardi deb aytish mumkin emas.
Yechim: barcha boshqaruv qarorlari manbai logga yozilishi, AI perception hissasi o‘lchanishi va baseline alohida belgilanishi kerak.
20.6. Deadline va deployment
Murakkab CARLA + GPU + public web stack bir kun ichida tayyor bo‘lishi kafolatlanmaydi.
Yechim: P0 va P1 bosqichlarini mustaqil ishlaydigan qilib qurish, fallback tayyorlash.

21. HOZIRGI HOLAT
Kelishilgan:
DreamDrive AI g‘oyasi tanlangan.
Computer Vision trackga mos deb belgilangan.
Input video bo‘lishi kerak.
Ikki rejimli arxitektura tanlangan.
YOLOPv2 birlamchi perception nomzodi.
ByteTrack tracking nomzodi.
RF-DETR, Depth Anything va Qwen3-VL qo‘shimcha variantlar.
Navigation uchun A*/waypoints.
Decision uchun FSM.
Simulation control uchun PID/Pure Pursuit.
CARLA asosiy simulator nomzodi.
n8n core driving uchun kerak emas.
Public web demo yaratish rejalashtirilgan.
Hali bajarilgani tasdiqlanmagan:
GPU va CARLA compatibility testi.
Model weight installation.
Real video inference.
Model evaluation.
Frontend/backend implementation.
Closed-loop control.
Public deployment.
Test metrikalari.
Final submission.
Ushbu loyiha bo‘yicha hozirgi asosiy ish — to‘g‘ri texnik yechimni tanlab, P0'dan boshlab haqiqiy kod va test natijalarini yaratish.

22. FOYDALI RASMIY MANBALAR
CARLA: https://carla.org/
CARLA GitHub: https://github.com/carla-simulator/carla
CARLA sensor docs: https://carla.readthedocs.io/en/latest/ref_sensors/
YOLOPv2: https://github.com/CAIC-AD/YOLOPv2
RF-DETR: https://github.com/roboflow/rf-detr
ByteTrack: https://github.com/FoundationVision/ByteTrack
Depth Anything V2: https://github.com/DepthAnything/Depth-Anything-V2
Qwen3-VL-4B: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
HighwayEnv: https://highway-env.farama.org/
n8n: https://docs.n8n.io/
WIUT Hackathon: https://hackathon.wiut.uz/
Har bir modelning aniq checkpointi, commit/version raqami va litsenziyasi implementatsiya vaqtida qayd etilsin.

23. YANGI CHAT UCHUN ANIQ TOPSHIRIQ
Men WIUT Hackathon 2026 Computer Vision finalida DreamDrive AI loyihasini yaratmoqdaman.
Yuqoridagi barcha ma’lumotni loyiha spetsifikatsiyasi sifatida qabul qil.
Bizning asosiy maqsadimiz:
Haqiqiy video inputni qayta ishlaydigan Computer Vision tizimi va shu perception natijalaridan foydalanib virtual avtomobilni manzilga boshqaradigan, xavfsizlik monitoringiga ega AI agent prototipini yaratish.
Menga umumiy nazariy maslahatlar emas, amaliy engineering yordam kerak.
Quyidagi vazifalarni bajar:
Arxitekturani senior autonomous-driving ML engineer sifatida tanqidiy audit qil.
Ortiqcha modellarni olib tashla, minimal ishlaydigan stackni tanla.
Eng katta texnik xavflarni va birinchi tekshiriladigan shartlarni ko‘rsat.
Hardware cheklovlari uchun CARLA va yengil simulator variantlarini tayyorla.
Python backend va CV pipeline'ni real kod sifatida rejalashtir.
Pretrained weights va litsenziyalarni to‘g‘ri boshqar.
Video upload → inference → annotated result ishlaydigan birinchi vertical slice'ni yarat.
Keyin navigation, decision va virtual controller modulini integratsiya qil.
Perception va simulator ground truthni qat’iy ajrat.
Hakam yangi video berganda ishlaydigan testlarni yarat.
Metrikalarni faqat haqiqiy sinovlardan hisobla.
Public deployment, README va hackathon taqdimotini tayyorlashni rejalashtir.
Muhim talablar:
Fake demo yoki hard-coded inference bo‘lmasin.
To‘liq autonomous driving funksiyalari implementatsiya qilinmasdan turib, ularni tayyor deb ko‘rsatma.
Modelning noto‘g‘ri yoki ishonchsiz javoblarini yashirma.
Haqiqiy avtomobilni boshqarish yoki haydovchi uxlay olishini va’da qilma.
Barcha driving commands faqat simulyatorga uzatilsin.
Birinchi navbatda Computer Vision va end-to-end ishlaydigan MVP'ga e’tibor ber.
Kod modulli, hujjatlashtirilgan va test qilinadigan bo‘lsin.
Deadline: 11 October 2026, 13:00, Tashkent time.
Endi shu loyiha asosida implementatsiyani boshlash uchun eng optimal texnik qarorlar va ishning birinchi aniq bosqichini ishlab chiq.
Hujjat bo‘yicha ikkita muhim eslatma
Birinchisi, DreamDrive AI hozircha ishchi nom. Lucid avtomobil kompaniyasi ham DreamDrive nomidan o‘z haydovchiga yordam tizimi uchun foydalanadi. Shuning uchun hackathondan keyingi brendlash jarayonida savdo belgisi va nom bandligini tekshirish kerak.
Ikkinchisi, texnik jihatdan eng katta qiyinchilik modelni ishga tushirish emas, balki Computer Vision natijalarini virtual avtomobilning haqiqiy boshqaruv sikliga ulash. Yangi chatda ishni aynan shu integratsiyaning kichik, o‘lchanadigan versiyasidan boshlashni tavsiya qilaman.
Hujjatdagi CARLA, RF-DETR va n8n imkoniyatlari bo‘yicha tavsiyalar ularning amaldagi hujjatlariga asoslangan. 

GitHub
+2
Shu master-hujjat yangi chatga loyihaning hozirgi maqsadi, texnik tanlovlari, cheklovlari va bajarilmagan ishlarini yetkazish uchun yetarli. 
