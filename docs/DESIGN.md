# Product interface specification

Build a premium, restrained automotive research dashboard. The primary camera canvas anchors the page. Use light neutral backgrounds, white surfaces, crisp navy text and blue/cyan accents. A dark video panel is acceptable; avoid an all-dark cluttered admin template. No random gradients, emoji controls, giant decorative text or invented KPIs.

Tokens: background #F4F7FB; surface #FFFFFF; text #12233B; muted #63748A; accent #2563EB; cyan #0891B2; border #E0E7EF; success #15803D; warning #B45309; danger #B91C1C. Consistent 8px spacing scale, 12–16px radii, subtle elevation, 1px borders. Use a locally served readable sans font with proper Uzbek apostrophes. Headings 24–32px, body 14–16px. Tabs and clear labels reduce visual noise.

Desktop: 232px navigation rail, compact top bar (project/run/build/capability), main content with 8/4 split camera and side decision panel; below, timeline and results. Video aspect ratio fixed without stretching. Sidebar: Umumiy, Video tahlili, Simulyatsiya, Baholash, Tizim. Mobile: collapsible navigation, camera first, controls next, horizontally scrollable timeline/table with labeled content.

## Video page
Visible upload dropzone and supported limits; real progress, elapsed time and queue state. Working original/annotated toggle; overlays with model/source labels; timestamp scrubber synchronizes event selection. Result area shows analyzed frames, measured latency distribution, tracks and report downloads. No empty metric is zero by default. Missing weights and failed decode have actionable explanations.

## Simulation page
Virtual graph map with actual selectable nodes, selected route and moving ego pose. Live RGB camera with overlay/source pill. Start/pause/resume/stop controls reflect backend states; stop is always reachable and remains labeled stopping until confirmed. A panel shows current FSM state, short reason, steering/throttle/brake, ego speed and acquisition health. Small provenance legend: ML inference, synthetic pixel CV, telemetry and evaluation truth. Do not call synthetic CV “trained AI vision”.

## Evaluation and system pages
Compare runs by commit/model/data/source/device/config, real sample counts and evidence. Performance graph shows raw measured data windows. System readiness and limitations distinguish absent features from offline services. All controls either work or are disabled with an explanation. No fake map API or fabricated location search; MVP destinations are virtual map nodes.

Every async action has loading/empty/error/cancelled/retry states. Provide keyboard navigation, visible focus, aria labels, usable contrast and reduced-motion preference. Responsive screenshots at 1440x900, 768x1024 and 390x844. QA must inspect camera alignment, translated text wrapping, side panels, timeline and long errors, plus no console errors or network calls to host-only localhost.
