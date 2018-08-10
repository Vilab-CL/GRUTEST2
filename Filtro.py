import datetime 
file = open("weather-denmark.csv", "r").readlines()
file.pop(0)
dias = {}
for line in  file:
	datos = line.replace("\n","").split(",")
	date = datos[0].split(" ")[0]
	print "%s,%s"%(datos[0],float(datos[1])+100)
	if  date not in dias.keys(): 
		dias[date] = float(datos[1])+100
	else:
		if dias[date] > (float(datos[1])+100 ) :
			dias[date] = float(datos[1])+100


# orden = dias.keys()
# orden.sort()
# ix =0


# fechas_de_heladas = []
# valores = [] 
# for i in range(len(orden)):
# 	dia = orden[i]
# 	if dias[dia] <= (-1+100) :
# 		ix = ix +1
# 		#print "HELADA",dia,dias[dia], "|",dias[orden[i-3]],dias[orden[i-2]],dias[orden[i-1]],dias[orden[i]]
# 		fechas_de_heladas.append(dia)
# 		valores.append(dias[orden[i-3]])
# 		valores.append(dias[orden[i-2]])
# 		valores.append(dias[orden[i-1]])
# 		valores.append(dias[orden[i]])



# #print "CANTIDAD DE HELADAS",ix,fechas_de_heladas
# #print "Valores",valores,len(valores)
# ix = 0 
# l = len(valores)
# print '"Date","Daily minimum temperatures in Melbourne, Australia, 1981-1990"'
# for i in range(len(orden)):
# 	dia = orden[i]
# 	fecha = datetime.datetime.strptime(dia, '%Y-%m-%d')
# 	out = ""
# 	if ( fecha.month >= 6 and fecha.month <= 11):
# 		out="%s"%(valores[ix%l])
# 		ix=ix+1
# 	else:
# 		ix = 0 
# 		out="%s"%(dias[dia])
# 	out = '"%s",%s'%(dia,out)
# 	#print out
# 	# for i in range(24):
# 	# 	hora = i 
# 	# 	if i < 10:
# 	# 		hora ="0%s"%(i)
# 	# 	print "%s %s:00,%s"%(dia,hora,out)



