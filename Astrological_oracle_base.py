from kerykeion import AstrologicalSubject, KerykeionChartSVG, Report
import sys
import os
import google.generativeai as genai
from io import StringIO

# Redefinir la función print para capturar la salida estándar
class Logger:
    def __init__(self):
        self.terminal = sys.stdout
        self.log = []

    def write(self, message):
        self.terminal.write(message)
        self.log.append(message)

    def flush(self):
        pass

# Reasignar stdout para capturar la salida
sys.stdout = Logger()

def generar_informe_persona(nombre, año, mes, día, hora, minutos, ciudad, nación=None):
    # Crear instancia para una persona
    persona = AstrologicalSubject(nombre, año, mes, día, hora, minutos, ciudad, nation=nación)
    
    # Generar reporte
    reporte = Report(persona)
    reporte.print_report()

    # Generar SVG para la carta astral individual
    svg_chart = KerykeionChartSVG(persona, chart_type="Natal")
    svg_chart.makeSVG()
    print("SVG generado correctamente")

    # Retornar la instancia para ser usada posteriormente
    return persona

def generar_imagen_sinastría(primera_persona, segunda_persona):
    # Generar SVG para sinastría
    svg_chart = KerykeionChartSVG(primera_persona, chart_type="Synastry", second_obj=segunda_persona)
    svg_chart.makeSVG()
    print("SVG de sinastría generado correctamente")

def procesar_respuesta_gemini(input_text, prompt):
    # Configurar GEMINI API
    genai.configure(api_key='api_key_google')  # Reemplazar con tu clave API de Gemini
    model = genai.GenerativeModel('gemini-pro')

    # Generar respuesta con GEMINI
    response = model.generate_content(f"{prompt}\n{input_text}", stream=False)
    full_response = ""
    for chunk in response:
        full_response += chunk.text

    return full_response

def main():
    opcion = input("¿Qué desea hacer?\n1. Generar informe de una persona\n2. Generar SVG de sinastría entre dos personas\nIngrese el número correspondiente: ")
    
    if opcion == "1":
        nombre = input("Ingrese el nombre de la persona: ")
        año = int(input("Ingrese el año de nacimiento: "))
        mes = int(input("Ingrese el mes de nacimiento (número): "))
        día = int(input("Ingrese el día de nacimiento: "))
        hora = int(input("Ingrese la hora de nacimiento (formato 24h): "))
        minutos = int(input("Ingrese los minutos de nacimiento: "))
        ciudad = input("Ingrese la ciudad de nacimiento: ")
        
        persona = generar_informe_persona(nombre, año, mes, día, hora, minutos, ciudad)

        # Exportar la salida capturada a un archivo de texto
        with open('informe_astrologico.txt', 'w') as f:
            for line in sys.stdout.log:
                f.write(line)

        # Leer el archivo de texto y procesar con GEMINI
        with open('informe_astrologico.txt', 'r') as f:
            informe_texto = f.read()
        
        prompt = "Actúa como astrólogo. Basándote en la información proporcionada, da un informe astrológico sobre las características generales de la persona segun sus astros y agrega secciones de salud, dinero y amor. Usa un estilo detallado y extenso, con al menos 2000 palabras."
        respuesta_gemini = procesar_respuesta_gemini(informe_texto, prompt)
        
        # Mostrar respuesta en la consola
        print(respuesta_gemini)

        # Guardar la respuesta en un nuevo archivo de texto
        with open(f'{nombre}_análisis_completo.txt', 'w') as f:
            f.write(respuesta_gemini)
    
    elif opcion == "2":
        # Primera persona
        nombre1 = input("Ingrese el nombre de la primera persona: ")
        año1 = int(input("Ingrese el año de nacimiento de la primera persona: "))
        mes1 = int(input("Ingrese el mes de nacimiento de la primera persona (número): "))
        día1 = int(input("Ingrese el día de nacimiento de la primera persona: "))
        hora1 = int(input("Ingrese la hora de nacimiento de la primera persona (formato 24h): "))
        minutos1 = int(input("Ingrese los minutos de nacimiento de la primera persona: "))
        ciudad1 = input("Ingrese la ciudad de nacimiento de la primera persona: ")
        
        primera_persona = generar_informe_persona(nombre1, año1, mes1, día1, hora1, minutos1, ciudad1)
        
        # Segunda persona
        nombre2 = input("Ingrese el nombre de la segunda persona: ")
        año2 = int(input("Ingrese el año de nacimiento de la segunda persona: "))
        mes2 = int(input("Ingrese el mes de nacimiento de la segunda persona (número): "))
        día2 = int(input("Ingrese el día de nacimiento de la segunda persona: "))
        hora2 = int(input("Ingrese la hora de nacimiento de la segunda persona (formato 24h): "))
        minutos2 = int(input("Ingrese los minutos de nacimiento de la segunda persona: "))
        ciudad2 = input("Ingrese la ciudad de nacimiento de la segunda persona: ")
        
        segunda_persona = generar_informe_persona(nombre2, año2, mes2, día2, hora2, minutos2, ciudad2)
        
        # Generar imagen de sinastría
        generar_imagen_sinastría(primera_persona, segunda_persona)

        # Exportar la salida capturada a un archivo de texto
        with open('informe_astrologico.txt', 'w') as f:
            for line in sys.stdout.log:
                f.write(line)

        # Leer el archivo de texto y procesar con GEMINI
        with open('informe_astrologico.txt', 'r') as f:
            informe_texto = f.read()
        
        prompt = "Actúa como astrólogo. Basándote en la información proporcionada, presenta un informe forense esotérico detallado sobre la compatibilidad de las personas mencionadas según sus cartas astrales. Incluye un análisis exhaustivo de la relación potencial y un porcentaje de compatibilidad basado en ello, con un mínimo de 2000 palabras."
        respuesta_gemini = procesar_respuesta_gemini(informe_texto, prompt)
        
        # Mostrar respuesta en la consola
        print(respuesta_gemini)

        # Guardar la respuesta en un nuevo archivo de texto
        with open(f'análisis_sinastría_{nombre1}_{nombre2}_análisis_completo.txt', 'w') as f:
            f.write(respuesta_gemini)
    
    else:
        print("Opción no válida. Por favor, ingrese 1 o 2.")

if __name__ == "__main__":
    main()
