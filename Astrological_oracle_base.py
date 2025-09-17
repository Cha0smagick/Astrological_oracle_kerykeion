#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script completo de astrología usando Kerykeion
Genera múltiples tipos de cartas astrales y reportes
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Optional, List, Dict, Any

from kerykeion import (
    AstrologicalSubject, 
    KerykeionChartSVG, 
    Report, 
    SynastryAspects,
    CompositeSubjectFactory
)


class KerykeionAdvanced:
    """Clase principal para gestionar todas las funcionalidades de Kerykeion"""
    
    def __init__(self):
        self.subjects = {}
        self.current_directory = os.getcwd()
        
    def create_subject(self, name: str, year: int, month: int, day: int, 
                      hour: int, minute: int, city: str, country: str,
                      lng: Optional[float] = None, 
                      lat: Optional[float] = None,
                      tz_str: Optional[str] = None,
                      zodiac_type: str = "Tropical",
                      sidereal_mode: str = "LAHIRI",
                      house_system: str = "P",  # Corregido: house_system en singular
                      perspective_type: str = "Geocentric") -> AstrologicalSubject:
        """
        Crea un sujeto astrológico con múltiples opciones de configuración
        """
        try:
            if lng and lat and tz_str:
                subject = AstrologicalSubject(
                    name, year, month, day, hour, minute,
                    lng=lng, lat=lat, tz_str=tz_str, city=city,
                    zodiac_type=zodiac_type,
                    sidereal_mode=sidereal_mode,
                    house_system=house_system,  # Corregido
                    perspective_type=perspective_type
                )
            else:
                subject = AstrologicalSubject(
                    name, year, month, day, hour, minute, city, country,
                    zodiac_type=zodiac_type,
                    sidereal_mode=sidereal_mode,
                    house_system=house_system,  # Corregido
                    perspective_type=perspective_type
                )
            
            self.subjects[name] = subject
            return subject
            
        except Exception as e:
            print(f"Error creando sujeto {name}: {e}")
            raise
    
    def create_output_directory(self, birth_date: datetime) -> str:
        """Crea directorio de resultados con fecha de nacimiento"""
        date_str = birth_date.strftime("%Y%m%d")
        dir_name = f"results_{date_str}"
        output_dir = os.path.join(self.current_directory, dir_name)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Directorio creado: {output_dir}")
        
        return output_dir
    
    def generate_birth_chart(self, subject: AstrologicalSubject, 
                           output_dir: str, 
                           theme: str = "classic",
                           language: str = "ES",
                           minify: bool = False,
                           remove_css_variables: bool = False) -> str:
        """Genera carta natal con múltiples opciones"""
        try:
            chart = KerykeionChartSVG(
                subject, 
                chart_type="Natal",
                chart_language=language,
                new_output_directory=output_dir,
                theme=theme
            )
            
            # Carta completa
            chart.makeSVG(
                minify=minify,
                remove_css_variables=remove_css_variables
            )
            
            # Solo rueda
            chart.makeWheelOnlySVG()
            
            return f"Carta natal generada para {subject.name}"
            
        except Exception as e:
            return f"Error generando carta natal: {e}"
    
    def generate_external_chart(self, subject: AstrologicalSubject,
                              output_dir: str,
                              theme: str = "classic",
                              language: str = "ES") -> str:
        """Genera carta externa"""
        try:
            chart = KerykeionChartSVG(
                subject,
                chart_type="ExternalNatal",
                chart_language=language,
                new_output_directory=output_dir,
                theme=theme
            )
            
            chart.makeSVG()
            chart.makeWheelOnlySVG(wheel_only=True, wheel_only_external=True)
            
            return f"Carta externa generada para {subject.name}"
            
        except Exception as e:
            return f"Error generando carta externa: {e}"
    
    def generate_synastry_chart(self, subject1: AstrologicalSubject,
                              subject2: AstrologicalSubject,
                              output_dir: str,
                              theme: str = "classic",
                              language: str = "ES") -> str:
        """Genera carta de sinastría"""
        try:
            chart = KerykeionChartSVG(
                subject1, "Synastry", subject2,
                chart_language=language,
                new_output_directory=output_dir,
                theme=theme
            )
            
            chart.makeSVG()
            chart.makeWheelOnlySVG()
            
            # Aspectos de sinastría
            synastry = SynastryAspects(subject1, subject2)
            aspects = synastry.get_relevant_aspects()
            
            # Guardar aspectos en archivo
            aspects_file = os.path.join(output_dir, f"synastry_{subject1.name}_{subject2.name}.txt")
            with open(aspects_file, 'w', encoding='utf-8') as f:
                f.write(f"ASPECTOS DE SINASTRÍA: {subject1.name} & {subject2.name}\n")
                f.write("=" * 60 + "\n\n")
                
                for aspect in aspects:
                    f.write(f"{aspect['p1_name']} {aspect['aspect']} {aspect['p2_name']}: "
                           f"Orbe {aspect['orbit']:.2f}°, Diferencia {aspect['diff']:.2f}°\n")
            
            return f"Sinastría generada entre {subject1.name} y {subject2.name}"
            
        except Exception as e:
            return f"Error generando sinastría: {e}"
    
    def generate_transit_chart(self, natal_subject: AstrologicalSubject,
                             transit_date: datetime,
                             output_dir: str,
                             theme: str = "classic",
                             language: str = "ES") -> str:
        """Genera carta de tránsitos"""
        try:
            transit_subject = AstrologicalSubject(
                "Transits",
                transit_date.year,
                transit_date.month,
                transit_date.day,
                12, 0,  # Hora del mediodía
                natal_subject.city,
                natal_subject.nation
            )
            
            chart = KerykeionChartSVG(
                natal_subject, "Transit", transit_subject,
                chart_language=language,
                new_output_directory=output_dir,
                theme=theme
            )
            
            chart.makeSVG()
            
            return f"Tránsitos generados para {natal_subject.name}"
            
        except Exception as e:
            return f"Error generando tránsitos: {e}"
    
    def generate_composite_chart(self, subject1: AstrologicalSubject,
                               subject2: AstrologicalSubject,
                               output_dir: str,
                               theme: str = "classic",
                               language: str = "ES") -> str:
        """Genera carta compuesta"""
        try:
            factory = CompositeSubjectFactory(subject1, subject2)
            composite = factory.get_midpoint_composite_subject_model()
            
            chart = KerykeionChartSVG(
                composite, "Composite",
                chart_language=language,
                new_output_directory=output_dir,
                theme=theme
            )
            
            chart.makeSVG()
            
            return f"Carta compuesta generada entre {subject1.name} y {subject2.name}"
            
        except Exception as e:
            return f"Error generando carta compuesta: {e}"
    
    def generate_report(self, subject: AstrologicalSubject, output_dir: str) -> str:
        """Genera reporte completo"""
        try:
            report = Report(subject)
            
            # Guardar reporte en archivo
            report_file = os.path.join(output_dir, f"report_{subject.name}.txt")
            with open(report_file, 'w', encoding='utf-8') as f:
                # Redirigir stdout temporalmente
                original_stdout = sys.stdout
                sys.stdout = f
                report.print_report()
                sys.stdout = original_stdout
            
            return f"Reporte generado para {subject.name}"
            
        except Exception as e:
            return f"Error generando reporte: {e}"
    
    def generate_all_charts(self, subject: AstrologicalSubject, output_dir: str):
        """Genera todos los tipos de cartas para un sujeto"""
        results = []
        
        # Carta natal
        results.append(self.generate_birth_chart(subject, output_dir))
        
        # Carta externa
        results.append(self.generate_external_chart(subject, output_dir))
        
        # Reporte
        results.append(self.generate_report(subject, output_dir))
        
        # Tránsitos actuales
        current_date = datetime.now()
        results.append(self.generate_transit_chart(subject, current_date, output_dir))
        
        return results


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Generador avanzado de cartas astrales con Kerykeion")
    
    # Argumentos básicos
    parser.add_argument("--name", required=True, help="Nombre de la persona")
    parser.add_argument("--year", type=int, required=True, help="Año de nacimiento")
    parser.add_argument("--month", type=int, required=True, help="Mes de nacimiento")
    parser.add_argument("--day", type=int, required=True, help="Día de nacimiento")
    parser.add_argument("--hour", type=int, required=True, help="Hora de nacimiento")
    parser.add_argument("--minute", type=int, required=True, help="Minuto de nacimiento")
    parser.add_argument("--city", required=True, help="Ciudad de nacimiento")
    parser.add_argument("--country", required=True, help="País de nacimiento")
    
    # Argumentos opcionales
    parser.add_argument("--lng", type=float, help="Longitud (opcional)")
    parser.add_argument("--lat", type=float, help="Latitud (opcional)")
    parser.add_argument("--tz", help="Zona horaria (ej. Europe/Madrid)")
    parser.add_argument("--zodiac", default="Tropical", help="Tipo de zodiaco: Tropical/Sidereal")
    parser.add_argument("--theme", default="classic", help="Tema: classic/dark/light")
    parser.add_argument("--language", default="ES", help="Idioma: ES/EN/FR/PT/IT/DE")
    parser.add_argument("--house-system", default="P", help="Sistema de casas: P (Placidus), K (Koch), etc.")
    
    args = parser.parse_args()
    
    # Crear instancia principal
    kerykeion = KerykeionAdvanced()
    
    try:
        # Crear sujeto astrológico
        birth_date = datetime(args.year, args.month, args.day)
        subject = kerykeion.create_subject(
            name=args.name,
            year=args.year,
            month=args.month,
            day=args.day,
            hour=args.hour,
            minute=args.minute,
            city=args.city,
            country=args.country,
            lng=args.lng,
            lat=args.lat,
            tz_str=args.tz,
            zodiac_type=args.zodiac,
            house_system=getattr(args, 'house_system', 'P')  # Corregido
        )
        
        # Crear directorio de resultados
        output_dir = kerykeion.create_output_directory(birth_date)
        
        print(f"🧙‍♂️ Generando cartas astrales para {args.name}...")
        print(f"📁 Los resultados se guardarán en: {output_dir}")
        
        # Generar todas las cartas
        results = kerykeion.generate_all_charts(subject, output_dir)
        
        # Mostrar resultados
        print("\n" + "="*50)
        print("RESULTADOS DE LA GENERACIÓN:")
        print("="*50)
        for result in results:
            print(f"✅ {result}")
        
        print(f"\n🎉 ¡Proceso completado! Revisa los archivos en: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Ejemplo de uso directo (sin argumentos)
    if len(sys.argv) == 1:
        print("Ejecutando ejemplo por defecto...")
        
        kerykeion = KerykeionAdvanced()
        
        # Crear sujeto de ejemplo (John Lennon) - versión simplificada
        try:
            subject = AstrologicalSubject(
                "John Lennon",
                1940, 10, 9, 18, 30,
                "Liverpool", "GB"
            )
            
            # Crear directorio de resultados
            birth_date = datetime(1940, 10, 9)
            output_dir = kerykeion.create_output_directory(birth_date)
            
            # Generar cartas básicas
            print("Generando carta natal...")
            result1 = kerykeion.generate_birth_chart(subject, output_dir)
            print(f"✅ {result1}")
            
            print("Generando reporte...")
            result2 = kerykeion.generate_report(subject, output_dir)
            print(f"✅ {result2}")
            
            print(f"\n🎉 ¡Ejemplo completado! Revisa los archivos en: {output_dir}")
            
        except Exception as e:
            print(f"❌ Error en ejemplo: {e}")
            import traceback
            traceback.print_exc()
            
    else:
        main()
