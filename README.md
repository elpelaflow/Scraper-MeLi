# Web Scraping Mercado Livre

**Aviso legal:**
_Este es un proyecto personal utilizado únicamente con fines educativos/didácticos_

## Descripción general

Este proyecto utiliza la biblioteca **Scrapy** de Python para realizar web scraping en Mercado Livre, recopilando específicamente información sobre **bajos de 5 cuerdas**.

## Adaptación para otros artículos

¡Si deseas extraer datos para otro artículo, es totalmente posible!

### 1. Ve al archivo ubicado en

```bash
extraction/spiders/mercadolivre.py
```

### 2. Actualiza la URL inicial

Configúrala con el artículo que desees extraer.
Si quieres obtener precios de notebooks Acer en el sitio argentino, la URL sería

```bash
https://listado.mercadolibre.com.ar/notebook-acer
```

### 3. Actualiza la función parse

Haz clic en el botón «Siguiente página» y observa la nueva URL. En la versión argentina debería verse así:

```bash
https://listado.mercadolibre.com.ar/computacion/notebooks-accesorios/notebooks/acer/notebook-acer_Desde_49_NoIndex_True
```

Define esta URL como el atributo _next page_ en la clase MercadoLivreSpider, pero cambia _49_ por {offset}.

Esto garantizará que el crawler avance por las siguientes páginas.

Al final, el código del atributo _next page_ debería verse así:

```bash
next_page = f"https://listado.mercadolibre.com.ar/instrumentos-musicales/instrumentos-de-cuerda/bajos/bajo-5-cuerdas_Desde_{offset}_NoIndex_True"
```

Ten presente que en Mercado Libre Argentina las URL utilizan español rioplatense, por lo que los segmentos de ruta pueden variar (por ejemplo, `instrumentos-musicales` en lugar de `instrumentos-musicais`). Ajusta los valores según el artículo que estés rastreando.

> ℹ️ Nota: El sitio argentino muestra los precios en **pesos argentinos (ARS)** y la interfaz está en **español**. Verifica que tu pipeline de transformación y visualización maneje correctamente la moneda y, si es necesario, la traducción de campos.

### Dashboard

Actualmente, el dashboard se ve así:

!['Screenshot of the dashboard'](assets/screenshot.png)

Puedes buscar todos los artículos y aplicar filtros.

## Cómo instalar y ejecutar el proyecto

### Con Docker

He refactorizado el proyecto para ofrecer compatibilidad con Docker.

Puedes instalarlo con los siguientes comandos:

```bash
docker build -t mlscrape .
```

```bash
docker run -p 8501:8501 mlscrape
```

Esto asignará tu puerto 8501 al expuesto en el Dockerfile.
