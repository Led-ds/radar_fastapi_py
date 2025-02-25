# API de análise de deformação

Este é um projeto FastAPI para analisar a deformação de imagens usando um modelo de aprendizado de máquina treinado.

## 🚀 Início rápido

1. **Crie um ambiente virtual**:
```bash
python -m venv venv
source venv/bin/activate # No Windows, use: venv\Scripts\activate
```

2. **Instale dependências**:
```bash
pip install -r requirements.txt
```

3. **Inicie o servidor**:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8082 --reload
```

4. **Teste a API**:
- Abra seu navegador e acesse `http://localhost:8082/docs`
- Use a interface do usuário do Swagger para fazer upload de uma imagem e ver a previsão.