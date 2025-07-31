import streamlit as st
from kerykeion import AstrologicalSubject, KerykeionChartSVG, SynastryAspects
from kerykeion.report import Report
import sys
import os
import re
import google.generativeai as genai
from io import StringIO

# Función para limpiar texto para Google GEMINI
def clean_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return cleaned_text

# Función para generar respuesta desde Google GEMINI
def generate_response(cleaned_input, model):
    try:
        response = model.generate_content(cleaned_input, stream=True)
        full_response = ""
        for chunk in response:
            full_response += chunk.text
        return full_response
    except Exception as e:
        error_message = str(e)
        st.error(f"Error: {error_message}")
        return None

# Función para obtener la entrada del usuario para las cartas astrológicas, con claves únicas
def get_user_input(key_suffix=""):
    name = st.text_input(f"Nombre de la persona {key_suffix}:", key=f"name{key_suffix}")
    year = st.number_input(f"Año de nacimiento {key_suffix}:", min_value=1900, max_value=2099, key=f"year{key_suffix}")
    month = st.number_input(f"Mes de nacimiento {key_suffix} (1-12):", min_value=1, max_value=12, key=f"month{key_suffix}")
    day = st.number_input(f"Día de nacimiento {key_suffix} (1-31):", min_value=1, max_value=31, key=f"day{key_suffix}")
    hour = st.number_input(f"Hora de nacimiento {key_suffix} (0-23):", min_value=0, max_value=23, key=f"hour{key_suffix}")
    minute = st.number_input(f"Minuto de nacimiento {key_suffix} (0-59):", min_value=0, max_value=59, key=f"minute{key_suffix}")
    location = st.text_input(f"Lugar de nacimiento {key_suffix}:", key=f"location{key_suffix}")
    zodiac_type = st.selectbox(f"Tipo de zodiaco {key_suffix}", ["Tropic", "Sidereal"], key=f"zodiac_type{key_suffix}").capitalize()
    return name, year, month, day, hour, minute, location, zodiac_type

class SynastryAspectsWithRelevant(SynastryAspects):
    @property
    def relevant_aspects(self):
        relevant_aspects_list = []
        
        # Agrega tu lógica para calcular aspectos relevantes aquí
        for aspect in self.all_aspects:
            # Personaliza esta condición según tus criterios
            if aspect["orbit"] < 10:
                relevant_aspects_list.append(aspect)
        
        return relevant_aspects_list

def generate_astrological_report(person):
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    user_report = Report(person)
    user_report.print_report()
    sys.stdout = old_stdout
    return mystdout.getvalue()

def main():
    st.title("Oráculo de Cartas Astrológicas")
    st.markdown(
    """
    **Autor:** Cha0smagick el Tecnomago  
    **Creado para el blog:** [El Rincón Paranormal](https://elrinconparanormal.blogspot.com)  
    **Página principal del proyecto:** [Cha0smagick's Astrological oracle](https://elrinconparanormal.blogspot.com/2023/12/Free%20Astrological%20Chart%20Generator%20Tool%20Online.html)
    """
    )

    # Generación de cartas astrológicas
    st.write("Ingrese la información para la primera persona:")
    name1, year1, month1, day1, hour1, minute1, location1, zodiac_type1 = get_user_input()
    person1 = AstrologicalSubject(name1, year1, month1, day1, hour1, minute1, location1, zodiac_type=zodiac_type1)

    st.write("Ingrese la información para la segunda persona:")
    name2, year2, month2, day2, hour2, minute2, location2, zodiac_type2 = get_user_input(" - Persona 2")
    person2 = AstrologicalSubject(name2, year2, month2, day2, hour2, minute2, location2, zodiac_type=zodiac_type2)

    chart_type = st.selectbox("Tipo de carta", ["Natal", "Sinastría", "Tránsito"]).capitalize()

    report_content_person1 = generate_astrological_report(person1)
    report_content_person2 = generate_astrological_report(person2)
    combined_report_content = report_content_person1 + "\n" + report_content_person2

    if chart_type == "Sinastía":
        synastry = SynastryAspectsWithRelevant(person1, person2)
        aspect_list = synastry.relevant_aspects
        combined_report_content += "\n" + "\n".join([str(aspect) for aspect in aspect_list])
    elif chart_type == "Tránsito":
        st.write("Característica en desarrollo")

    # Integración con Google GEMINI
    genai.configure(api_key='YOUR_API_KEY')  # Reemplaza con tu clave de API de Gemini
    model = genai.GenerativeModel('gemini-pro')

    st.write("Consulta al Oráculo Astrológico usando la información de tu carta astrológica como contexto")
    user_query = st.text_input("Tu pregunta:")
    if st.button("Obtener visión astrológica"):
        cleaned_input = clean_text(combined_report_content + " " + user_query)
        response = generate_response(cleaned_input, model)
        if response:
            st.success(response)

    # Mostrar las cartas astrológicas generadas
    st.markdown("## Cartas Astrológicas Generadas")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    svg_files = [f for f in os.listdir(script_dir) if f.endswith(".svg")]
    if svg_files:
        for svg_file in svg_files:
            st.image(os.path.join(script_dir, svg_file), use_container_width=True)
    else:
        st.write("No se encontraron archivos SVG en el directorio actual.")

if __name__ == "__main__":
    main()
