package org.jabe.neverland.ssl;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.net.Socket;
import java.security.KeyStore;

import javax.net.ServerSocketFactory;
import javax.net.ssl.KeyManagerFactory;
import javax.net.ssl.SSLContext;
import javax.net.ssl.SSLServerSocket;

public class SSLServer extends Thread {
	
	private Socket socket;
	
	public SSLServer(Socket socket) {  
        this.socket = socket;  
    } 

	public void run() {
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(
					socket.getInputStream()));
			System.out.println(reader.readLine());
			reader.close();
			socket.close();
		} catch (IOException e) {
			
		}
	}

	private static String SERVER_KEY_STORE = "src\\org\\jabe\\neverland\\ssl\\server_ks";
	
	private static String SERVER_KEY_STORE_PASSWORD = "server";

	public static void main(String[] args) throws Exception {
		System.setProperty("javax.net.ssl.trustStore", SERVER_KEY_STORE);
		SSLContext context = SSLContext.getInstance("TLS");

		KeyStore ks = KeyStore.getInstance("jceks");
		ks.load(new FileInputStream(SERVER_KEY_STORE), null);
		KeyManagerFactory kf = KeyManagerFactory.getInstance("SunX509");
		kf.init(ks, SERVER_KEY_STORE_PASSWORD.toCharArray());

		context.init(kf.getKeyManagers(), null, null);

		ServerSocketFactory factory = context.getServerSocketFactory();
		ServerSocket _socket = factory.createServerSocket(8443);
		((SSLServerSocket) _socket).setNeedClientAuth(false);

		while (true) {
			new SSLServer(_socket.accept()).start();
		}
	}

}
