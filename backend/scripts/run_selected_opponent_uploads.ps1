$files = @(
  "opponent-team-1778126911266-a04929c1.jpg",
  "opponent-team-1778127403719-7675fd84.jpg",
  "opponent-team-1778213582512-6c42ee96.jpg",
  "opponent-team-1778214001673-59b8981e.jpg",
  "opponent-team-1778214871767-07e17c15.jpg",
  "opponent-team-1778214874250-1f045dd7.jpg",
  "opponent-team-1778215349515-d6110e2e.jpg",
  "opponent-team-1778216349821-132e55ba.jpg",
  "opponent-team-1778216761693-cdf76bd1.jpg",
  "opponent-team-1778217284721-d4a07ec5.jpg",
  "opponent-team-1778217288315-69f46cd6.jpg",
  "opponent-team-1778217299799-4095446f.jpg",
  "opponent-team-1778217799727-470eb4f9.jpg",
  "opponent-team-1778218388102-e510b3d4.jpg",
  "opponent-team-1778218389921-9fa32bdc.jpg",
  "opponent-team-1778218395324-b3391a20.jpg"
)

foreach ($file in $files) {
  $stem = [IO.Path]::GetFileNameWithoutExtension($file)
  .\.venv\Scripts\python.exe .\backend\scripts\cv_runtime\run_opponent_detection.py --image "uploads\$file" --output "cv\debug_reports\$stem.json"
}
