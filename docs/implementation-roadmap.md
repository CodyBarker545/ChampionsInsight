# Champions Insight Implementation Roadmap

## Milestone 1: App Skeleton

- React frontend with team input controls
- Flask backend with health and mock analysis endpoints
- End-to-end request from frontend to backend

## Milestone 2: Local Data Lookup

- Define champion data schema
- Add lookup endpoint
- Replace mock names and stats with dataset values

## Milestone 3: Calculators

- Implement speed comparison
- Implement first damage range calculator
- Add validation for missing or invalid inputs

## Milestone 4: Opponent Detection

- Add image upload endpoint
- Store fixed screen-region assumptions
- Return manual correction fields when detection confidence is low

## Milestone 5: RAG Explanations

- Add structured knowledge documents
- Retrieve mechanics context
- Return grounded answers or insufficient context
