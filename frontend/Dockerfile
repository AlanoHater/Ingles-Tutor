# ========================
# Solo para desarrollo local
# En producción → Vercel (no necesita este Dockerfile)
# ========================
FROM node:20-alpine

WORKDIR /app

# Instala dependencias primero (mejor cache)
COPY package.json package-lock.json* ./
RUN npm ci

# Copia el código
COPY . .

EXPOSE 3000

# Dev server con hot reload
CMD ["npm", "run", "dev"]