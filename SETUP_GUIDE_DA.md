# Trin-for-trin opsætning

## 1. Pak ZIP-filen ud
Du skal se `index.html`, `fetch_weather.py`, `requirements.txt`, `weather.json`, `README.md`, `SETUP_GUIDE_DA.md` og mappen `.github`.

## 2. Lav et nyt GitHub repository
GitHub -> + -> New repository. Navn fx `greenland-weather`. Vælg Public. Tilføj ikke README/.gitignore/license. Klik Create repository.

## 3. Upload de almindelige filer
Code -> Add file -> Upload files. Upload `index.html`, `fetch_weather.py`, `requirements.txt`, `weather.json`, `README.md`, `SETUP_GUIDE_DA.md`. Klik Commit changes.

## 4. Opret workflow-filen
Add file -> Create new file. Skriv præcis `.github/workflows/update-weather.yml` i filnavnsfeltet. Åbn den tilsvarende fil fra ZIP-mappen på din computer, kopier hele indholdet, indsæt i GitHub og Commit changes.

## 5. Tillad skriveadgang
Settings -> Actions -> General -> Workflow permissions -> Read and write permissions -> Save.

## 6. Indstil GitHub Pages
Settings -> Pages -> Build and deployment -> Source: `GitHub Actions`. Vælg IKKE Deploy from a branch i denne version.

## 7. Kør første opdatering
Actions -> `Update and publish weather dashboard` -> Run workflow -> Run workflow.

Workflowet prøver først THREDDS for SER_B, MIT_B og MIT. Hvis en station fejler dér, hentes den automatisk fra Dataverse. Derefter laves en samlet `weather.json` og dashboardet publiceres.

## 8. Find siden
Når workflowet er grønt: Settings -> Pages. Her vises adressen til dashboardet.

## 9. Hvad dashboardet viser
Tre knapper: SER_B, MIT_B, MIT. For valgt station vises aktuelle værdier, 7-dages min/max, 30-dages min/max, tre 30-dages grafer og kort.

## 10. Automatisk opdatering
Workflowet kører cirka én gang i timen ved minut 12. Browseren tjekker efter ny `weather.json` hvert 5. minut.

## 11. Hvis workflowet bliver rødt
Actions -> det røde run -> det røde trin. Kopiér teksten fra Traceback og ned.

## 12. Hvis data ser gamle ud
Åbn `weather.json` og se stationens `latest` -> `time`. Det er det seneste timestamp i den fil, GEUS faktisk leverede.
