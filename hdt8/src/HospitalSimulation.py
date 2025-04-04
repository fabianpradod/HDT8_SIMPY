import simpy
import random
import statistics
from dataclasses import dataclass
from collections import defaultdict

# Universidad del Valle de Guatemala
# Algoritmos y Estructuras de Datos
# Ing. Douglas Barrios
# @author: Fabian Prado, Joel Nerio
# Creacion: 03/04/2025
# Ultima modificacion: 03/04/2025
# File Name: HospitalSimulation.py
# Descripcion: Simulacion en SimPy sobre la atencion en un hospital promedio basado en informacion real
# Codigo base hecho con la ayuda de  Claude.ai

random.seed(10)

# Clase Patient
class Patient:
    def __init__(self, patientId, arrivalTime, triageCode):
        self.id = patientId
        self.arrivalTime = arrivalTime
        self.triageCode = triageCode
        self.priority = ord(triageCode) - ord('A') + 1
        self.guaranteedPriority = None
        self.times = {
            'arrivalTime': arrivalTime,
            'startTriage': None,
            'endTriage': None,
            'startDoctor': None,
            'endDoctor': None,
            'startXray': None,
            'endXray': None,
            'startLab': None,
            'endLab': None,
            'endTime': None
        }
        self.exams = {
            'xray': False,
            'lab': False
        }

# Clase Hospital
@dataclass
class Hospital:
    numNurses: int
    numDoctors: int
    numXrays: int
    numLabs: int
    simulationTime: int = 480
    arrivalInterval: float = 4.0

# Clase EmergencyService
class EmergencyService:
    def __init__(self, env, hospital):
        self.env = env
        self.hospital = hospital

        self.nurses = simpy.Resource(env, hospital.numNurses)
        self.doctors = simpy.PriorityResource(env, hospital.numDoctors)
        self.xrays = simpy.PriorityResource(env, hospital.numXrays)
        self.labs = simpy.PriorityResource(env, hospital.numLabs)

        self.attendedPatients = []
        self.waitingTimes = []
        self.patientsByService = {
            'xray': 0,
            'lab': 0
        }

    # Funcion: attendPatient
    # Descripcion: Procesa la atencion de un paciente a lo largo de las diferentes fases.
    def attendPatient(self, patient):
        patient.times['startTriage'] = self.env.now
        with self.nurses.request() as req:
            yield req
            triageTime = random.uniform(5, 15) # Tiempo promedio de una enfermera para asignar un triage
            yield self.env.timeout(triageTime)
            patient.times['endTriage'] = self.env.now

            if patient.guaranteedPriority is None:
                patient.priority = random.randint(1, 5)
            else:
                patient.priority = patient.guaranteedPriority

        yield self.env.timeout(random.uniform(1, 5))

        patient.times['startDoctor'] = self.env.now
        with self.doctors.request(priority=patient.priority) as req:
            yield req
            doctorTime = random.uniform(15, 45) # Tiempo promedio de atencion de un doctor al paciente
            yield self.env.timeout(doctorTime)
            patient.times['endDoctor'] = self.env.now

            needsXray = random.random() < 0.216 # Porcetaje promedio de gente que necesita un xray
            needsLab = random.random() < random.uniform(0.04, 0.09) # Porcentaje promedio de gente que necesita pruebas de lab

            if needsXray:
                yield self.env.timeout(random.uniform(1, 5))

                patient.times['startXray'] = self.env.now
                with self.xrays.request(priority=patient.priority) as reqXray:
                    yield reqXray
                    xrayTime = random.uniform(15, 30) # Tiempo promedio de xray
                    yield self.env.timeout(xrayTime)
                    patient.times['endXray'] = self.env.now
                    patient.exams['xray'] = True
                    self.patientsByService['xray'] += 1

                yield self.env.timeout(random.uniform(1, 5))

            if needsLab:
                yield self.env.timeout(random.uniform(1, 5))

                patient.times['startLab'] = self.env.now
                with self.labs.request(priority=patient.priority) as reqLab:
                    yield reqLab
                    labTime = random.uniform(30, 60) # Tiempo promedio de pruebas de laboratorio
                    yield self.env.timeout(labTime)
                    patient.times['endLab'] = self.env.now
                    patient.exams['lab'] = True
                    self.patientsByService['lab'] += 1

                yield self.env.timeout(random.uniform(1, 5))

            if needsXray or needsLab:
                yield self.env.timeout(random.uniform(10, 20))

        patient.times['endTime'] = self.env.now
        totalTime = patient.times['endTime'] - patient.arrivalTime
        self.attendedPatients.append(patient)
        self.waitingTimes.append(totalTime)

# Funcion: generatePatients
# Descripcion: Genera los pacientes y los envia al proceso de atencion en emergencias.
def generatePatients(env, emergencyService):
    patientId = 0
    requiredPriorities = list(range(1, 6))
    injectionTimeLimit = 300

    while True:
        # Inyeccion de pacientes con prioridad garantizada durante los primeros 300 minutos
        if env.now > 0 and env.now % 60 < 1 and env.now < injectionTimeLimit:
            existingPriorities = set(p.priority for p in emergencyService.attendedPatients)
            missingPriorities = [p for p in requiredPriorities if p not in existingPriorities]

            if missingPriorities:
                for priority in missingPriorities:
                    patient = Patient(patientId, env.now, chr(ord('A') + priority - 1))
                    patient.guaranteedPriority = priority
                    env.process(emergencyService.attendPatient(patient))
                    patientId += 1
                    yield env.timeout(0.5)

        code = chr(ord('A') + random.randint(0, 4))
        patient = Patient(patientId, env.now, code)
        env.process(emergencyService.attendPatient(patient))
        patientId += 1
        yield env.timeout(random.uniform(3, 5))

# Funcion: runSimulation
# Descripcion: Ejecuta la simulacion para una configuracion de hospital dada.
def runSimulation(configuration):
    env = simpy.Environment()
    emergencyService = EmergencyService(env, configuration)
    env.process(generatePatients(env, emergencyService))
    env.run(until=configuration.simulationTime)

    if emergencyService.waitingTimes:
        averageTime = statistics.mean(emergencyService.waitingTimes)
        stdDev = statistics.stdev(emergencyService.waitingTimes) if len(emergencyService.waitingTimes) > 1 else 0
    else:
        averageTime, stdDev = 0, 0

    xrayPercentage = (emergencyService.patientsByService['xray'] / len(emergencyService.attendedPatients) * 100) if emergencyService.attendedPatients else 0
    labPercentage = (emergencyService.patientsByService['lab'] / len(emergencyService.attendedPatients) * 100) if emergencyService.attendedPatients else 0

    return {
        'configuration': configuration,
        'attendedPatients': len(emergencyService.attendedPatients),
        'averageTime': averageTime,
        'stdDev': stdDev,
        'byPriority': calculateByPriority(emergencyService.attendedPatients),
        'xrayPercentage': xrayPercentage,
        'labPercentage': labPercentage
    }

# Funcion: calculateByPriority
# Descripcion: Calcula el numero de pacientes y tiempos promedio por prioridad.
def calculateByPriority(patients):
    results = defaultdict(lambda: {'cantidad': 0, 'averageTime': 0})

    for p in patients:
        results[p.priority]['cantidad'] += 1
        totalTime = p.times['endTime'] - p.arrivalTime
        results[p.priority]['totalTime'] = results[p.priority].get('totalTime', 0) + totalTime

        for phase in ['triage', 'doctor']:
            keyStart = 'start' + phase.capitalize()
            previousKey = 'arrivalTime' if phase == 'triage' else 'endTriage'
            waitTimePhase = p.times[keyStart] - p.times[previousKey]
            waitKey = f'wait{phase.capitalize()}Total'
            results[p.priority][waitKey] = results[p.priority].get(waitKey, 0) + waitTimePhase

    for priority in range(1, 6):
        if results[priority]['cantidad'] > 0:
            results[priority]['averageTime'] = (
                results[priority]['totalTime'] / results[priority]['cantidad']
            )
            for phase in ['triage', 'doctor']:
                waitKey = f'wait{phase.capitalize()}Total'
                if waitKey in results[priority]:
                    results[priority][f'wait{phase.capitalize()}Average'] = (
                        results[priority][waitKey] / results[priority]['cantidad']
                    )

    return results

# Funcion: analyzeCosts
# Descripcion: Analiza el costo por hora o uso de cada recurso del hospital.
def analyzeCosts(results):
    # Costo por hora de cada recurso
    nurseCost = 30  
    doctorCost = 95     
    xrayCost = 175    
    labCost = 150
    
    costAnalysis = []
    
    for res in results:
        config = res['configuration']
        simulationHours = config.simulationTime / 60
        
        personnelCost = (
            (config.numNurses * nurseCost * simulationHours) +
            (config.numDoctors * doctorCost * simulationHours)
        )
        
        equipmentCost = (
            (config.numXrays * xrayCost * simulationHours) +
            (config.numLabs * labCost * simulationHours)
        )
        
        totalCost = personnelCost + equipmentCost
        
        patientsPerHour = res['attendedPatients'] / simulationHours if simulationHours > 0 else 0
        costPerPatient = totalCost / res['attendedPatients'] if res['attendedPatients'] > 0 else 0
        costPerPatientPerHour = costPerPatient / simulationHours if simulationHours > 0 else 0
        
        resourceUtilization = res['attendedPatients'] / (
            config.numNurses + config.numDoctors + config.numXrays + config.numLabs
        )
        
        costData = {
            'config': config,
            'personnelCost': personnelCost,
            'equipmentCost': equipmentCost,
            'totalCost': totalCost,
            'costPerPatient': costPerPatient,
            'patientsPerHour': patientsPerHour,
            'costPerPatientPerHour': costPerPatientPerHour,
            'resourceUtilization': resourceUtilization,
            'simulationHours': simulationHours,
            'attendedPatients': res['attendedPatients'],
            'averageTime': res['averageTime']
        }
        
        costAnalysis.append(costData)
    
    return costAnalysis

# Funcion: printCostAnalysis
# Descripcion: Imprime el analisis de costos en un formato claro y organizado.
def printCostAnalysis(costAnalysis):
    
    print("\n" + "=" * 80)
    print("ANALISIS DE COSTOS Y EFICIENCIA".center(80))
    print("=" * 80)
    
    print("\nCOMPARACION DE CONFIGURACIONES:")
    
    for i, data in enumerate(costAnalysis, 1):
        print(f"\n--- Configuracion {i} ---")
        print(f"Pacientes atendidos: {data['attendedPatients']}")
        print(f"Costo por paciente: {data['costPerPatient']:.2f} GTQ")
        print(f"Pacientes por hora: {data['patientsPerHour']:.2f}")
        print(f"Costo por paciente por hora: {data['costPerPatientPerHour']:.2f} GTQ")
        print(f"Utilizacion de recursos: {data['resourceUtilization']:.2f}")
        print(f"Tiempo promedio: {data['averageTime']:.2f} min")
    
    print("\nDETALLES POR CONFIGURACION:")
    print("(asumiendo ingreso de GTQ 350 por paciente)")
    for i, data in enumerate(costAnalysis, 1):
        config = data['config']
        print(f"\n--- Configuracion {i} ---")
        print(f"Recursos: {config.numNurses} enfermeras, {config.numDoctors} doctores, "
              f"{config.numXrays} rayos X, {config.numLabs} laboratorios")
        print(f"Tiempo simulacion: {config.simulationTime} min ({data['simulationHours']:.2f} horas)")
        print(f"Costo personal: GTQ {data['personnelCost']:.2f}")
        print(f"Costo equipos: GTQ {data['equipmentCost']:.2f}")
        print(f"Costo total: GTQ {data['totalCost']:.2f}")
        print(f"Ganancia estimada: GTQ {data['attendedPatients'] * 350 - data['totalCost']:.2f} ")


# Funcion: enerateDataForGraphs
# Descripcion: Genera datos estructurados para graficas.
def generateDataForGraphs(results, costAnalysis):
    graphData = {
        'configLabels': [f"Config {i+1}" for i in range(len(results))],
        'patientCounts': [r['attendedPatients'] for r in results],
        'avgTimes': [r['averageTime'] for r in results],
        'costsPerPatient': [c['costPerPatient'] for c in costAnalysis],
        'patientsPerHour': [c['patientsPerHour'] for c in costAnalysis],
        'priorityDistribution': [],
        'waitTimesByPriority': [],
        'resourceCosts': []
    }
    
    # Datos de distribucion de prioridades
    for result in results:
        priorityData = {}
        for priority in range(1, 6):
            data = result['byPriority'][priority]
            priorityData[f"Prioridad {priority}"] = data['cantidad']
        graphData['priorityDistribution'].append(priorityData)
    
    # Datos de tiempos de espera por prioridad
    for result in results:
        waitTimeData = {}
        for priority in range(1, 6):
            data = result['byPriority'][priority]
            if data['cantidad'] > 0:
                waitTriage = data.get('waitTriageAverage', 0)
                waitDoctor = data.get('waitDoctorAverage', 0)
                waitTimeData[f"Prioridad {priority}"] = {
                    'triageWait': waitTriage,
                    'doctorWait': waitDoctor,
                    'totalTime': data['averageTime']
                }
        graphData['waitTimesByPriority'].append(waitTimeData)
    
    # Datos de costos de recursos
    for i, data in enumerate(costAnalysis):
        config = data['config']
        graphData['resourceCosts'].append({
            'nurses': config.numNurses * 30 * data['simulationHours'],
            'doctors': config.numDoctors * 95 * data['simulationHours'],
            'xray': config.numXrays * 175 * data['simulationHours'],
            'lab': config.numLabs * 150 * data['simulationHours']
        })
    
    return graphData

# Funcion: main
def main():
    hospitalConfigurations = [
        Hospital(numNurses=3, numDoctors=1, numXrays=1, numLabs=1),
        Hospital(numNurses=4, numDoctors=2, numXrays=1, numLabs=1),
        Hospital(numNurses=6, numDoctors=3, numXrays=2, numLabs=2),
        Hospital(numNurses=5, numDoctors=3, numXrays=2, numLabs=1, arrivalInterval=3),
        Hospital(numNurses=6, numDoctors=3, numXrays=2, numLabs=2, arrivalInterval=3, simulationTime=720),
    ]

    print("\n" + "=" * 80)
    print("SIMULACION DE SERVICIOS DE EMERGENCIA".center(80))
    print("=" * 80)
    
    print("\nConfiguraciones del Hospital a evaluar:")
    for i, config in enumerate(hospitalConfigurations, 1):
        print(f"\nConfiguracion {i}:")
        print(f"- Enfermeras: {config.numNurses}")
        print(f"- Doctores: {config.numDoctors}")
        print(f"- Rayos X: {config.numXrays}")
        print(f"- Laboratorios: {config.numLabs}")
        print(f"- Intervalo llegada: {config.arrivalInterval} min")
        print(f"- Tiempo simulacion: {config.simulationTime} min")


    results = []
    for config in hospitalConfigurations:
        result = runSimulation(config)
        results.append(result)
    

    print("\n" + "=" * 80)
    print("RESUMEN GENERAL DE RESULTADOS".center(80))
    print("=" * 80)

    print("\nCOMPARACION DE RESULTADOS:")

    for i, result in enumerate(results, 1):
        print(f"\n--- Configuracion {i} ---")
        print(f"Pacientes atendidos: {result['attendedPatients']}")
        print(f"Tiempo promedio: {result['averageTime']:.2f} min")
        print(f"Desviacion estandar: {result['stdDev']:.2f} min")
        print(f"Porcentaje de rayos X: {result['xrayPercentage']:.2f}%")
        print(f"Porcentaje de laboratorio: {result['labPercentage']:.2f}%")
    
    
    # Analisis de costos
    costAnalysis = analyzeCosts(results)
    printCostAnalysis(costAnalysis)
    
    graphData = generateDataForGraphs(results, costAnalysis)
    
    return {
        'results': results,
        'costAnalysis': costAnalysis,
        'graphData': graphData
    }


# Funcion: generateGraphs
# Descripcion: Funcion para generar visualizaciones
def generateGraphs(data):
    import matplotlib.pyplot as plt
    import numpy as np
    
    graphData = data['graphData']
    results = data['results']
    costAnalysis = data['costAnalysis']
    
    plt.style.use('ggplot')
    
    # 1. Grafico principal
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(graphData['configLabels']))
    width = 0.2
    
    bar1 = ax.bar(x - width, graphData['patientCounts'], width, label='Pacientes Atendidos')
    
    bar2 = ax.bar(x, graphData['avgTimes'], width, label='Tiempo Promedio (min)')
    
    bar3 = ax.bar(x + width, [c/10 for c in graphData['costsPerPatient']], width, 
                  label='Costo por Paciente / 10 (GTQ)')
    
    ax.set_xticks(x)
    ax.set_xticklabels(graphData['configLabels'])
    ax.set_ylabel('Valores')
    ax.set_title('Comparativa entre Configuraciones')
    ax.legend()
    
    for bar in [bar1, bar2, bar3]:
        for rect in bar:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    # 2. Grafico de distribucion de prioridades 
    num_configs = len(results)
    fig, axes = plt.subplots(1, num_configs, figsize=(6 * num_configs, 5))

    if num_configs == 1:
        axes = [axes]

    for i, (ax, priorityData) in enumerate(zip(axes, graphData['priorityDistribution'])):
        priorities = list(priorityData.keys())
        counts = list(priorityData.values())
        ax.pie(counts, labels=priorities, autopct='%1.1f%%', startangle=90)
        ax.set_title(f'Config {i+1}', pad=20)

    plt.subplots_adjust(wspace=0.4)
    plt.tight_layout(pad=3.0)
    
    # 3. Grafico de eficiencia de costos
    fig, ax = plt.subplots(figsize=(10, 6))
    
    costEfficiency = []
    for data in costAnalysis:
        costEfficiency.append({
            'config': data['config'],
            'costPerPatient': data['costPerPatient'],
            'patientsPerHour': data['patientsPerHour']
        })
    
    # Ordenar costo por paciente
    sortedEfficiency = sorted(costEfficiency, key=lambda x: x['costPerPatient'])
    
    configNames = [f"Config {i+1}" for i, _ in enumerate(sortedEfficiency)]
    costs = [data['costPerPatient'] for data in sortedEfficiency]
    throughputs = [data['patientsPerHour'] for data in sortedEfficiency]
    
    x = np.arange(len(configNames))
    width = 0.35
    
    ax.bar(x - width/2, costs, width, label='Costo por Paciente (GTQ)')
    ax2 = ax.twinx()
    ax2.bar(x + width/2, throughputs, width, color='orange', label='Pacientes por Hora')
    
    ax.set_xticks(x)
    ax.set_xticklabels(configNames)
    ax.set_ylabel('Costo por Paciente (GTQ)')
    ax2.set_ylabel('Pacientes por Hora')
    ax.set_title('Eficiencia de Costos por Configuracion')
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # 4. Grafico de costos
    fig, ax = plt.subplots(figsize=(12, 6))
    
    nurseCosts = [data['nurses'] for data in graphData['resourceCosts']]
    doctorCosts = [data['doctors'] for data in graphData['resourceCosts']]
    xrayCosts = [data['xray'] for data in graphData['resourceCosts']]
    labCosts = [data['lab'] for data in graphData['resourceCosts']]
    
    bottom = np.zeros(len(graphData['configLabels']))
    
    p1 = ax.bar(graphData['configLabels'], nurseCosts, label='Enfermeras')
    bottom = np.add(bottom, nurseCosts)
    
    p2 = ax.bar(graphData['configLabels'], doctorCosts, bottom=bottom, label='Doctores')
    bottom = np.add(bottom, doctorCosts)
    
    p3 = ax.bar(graphData['configLabels'], xrayCosts, bottom=bottom, label='Rayos X')
    bottom = np.add(bottom, xrayCosts)
    
    p4 = ax.bar(graphData['configLabels'], labCosts, bottom=bottom, label='Laboratorios')
    
    ax.set_ylabel('Costo (GTQ)')
    ax.set_title('Desglose de Costos por Configuracion')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig

if __name__ == '__main__':
    simulationData = main()
    generateGraphs(simulationData)

