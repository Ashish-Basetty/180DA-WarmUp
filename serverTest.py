import socket
serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Assigns a port for the server that listens to clients connection to this port.
serv.bind(('0.0.0.0', 8080))
serv.listen(5)
while True:
	conn, addr = serv.accept()
	from_client = ''
	while True:
		conn, addr = serv.accept()
		if not data: break
		from_client += data.decode('utf_8')
		print(from_client)
		conn.sen("I am SERVER\n".encode())
	conn.close()
	print('client disconnected')
