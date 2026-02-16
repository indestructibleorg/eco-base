# Ecosystem Project Structure

```
ecosystem/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
├── client/
│   ├── public/
│   └── src/
├── server/
│   └── _core/
├── shared/
│   └── _core/
├── infrastructure/
│   ├── kustomize/
│   │   ├── base/
│   │   │   ├── namespace.yaml
│   │   │   ├── client-deployment.yaml
│   │   │   ├── server-deployment.yaml
│   │   │   ├── ingress.yaml
│   │   │   └── kustomization.yaml
│   │   └── overlays/
│   │       ├── staging/
│   │       │   ├── kustomization.yaml
│   │       │   ├── replicas-patch.yaml
│   │       │   └── ingress.yaml
│   │       └── production/
│   │           ├── kustomization.yaml
│   │           ├── replicas-patch.yaml
│   │           └── ingress.yaml
│   └── terraform/
├── ci-cd/
│   └── github-actions/
├── docs/
│   └── deployment/
├── drizzle/
│   ├── migrations/
│   └── meta/
├── patches/
├── outputs/
├── package.json
├── pnpm-lock.yaml
├── tsconfig.json
├── vite.config.ts
├── .env.example
└── README.md
```