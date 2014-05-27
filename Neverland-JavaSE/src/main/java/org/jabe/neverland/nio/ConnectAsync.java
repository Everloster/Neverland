package org.jabe.neverland.nio;

import java.nio.ByteBuffer;
import java.nio.channels.SocketChannel;
import java.net.InetSocketAddress;

/**
 * Demonstrate asynchronous connection of a SocketChannel.
 * 
 * @author Ron Hitchens (ron@ronsoft.com)
 */
public class ConnectAsync {
	public static void main(String[] argv) throws Exception {
		String host = "localhost";
		int port = 12340;
		if (argv.length == 2) {
			host = argv[0];
			port = Integer.parseInt(argv[1]);
		}
		InetSocketAddress addr = new InetSocketAddress(host, port);
		SocketChannel sc = SocketChannel.open();
		sc.configureBlocking(false);
		System.out.println("initiating connection");
		sc.connect(addr);
		while (!sc.finishConnect()) {
			doSomethingUseful();
		}
		System.out.println("connection established");
		// Do something with the connected socket
		// The SocketChannel is still nonblocking
		ByteBuffer temp = ByteBuffer.allocate(100);
		temp.put("I am jabe, I love ayamana!".getBytes());
		temp.flip();
		sc.write(temp);
		
		Thread.sleep(4000);
		
		temp.clear();
		sc.read(temp);
		temp.flip();
		dumpBuffer("response", temp);
		sc.close();
	}

	private static void doSomethingUseful() {
		System.out.println("doing something useless");
	}

	// Dump buffer content, counting and skipping nulls
	public static void dumpBuffer(String prefix, ByteBuffer buffer)
			throws Exception {
		System.out.print(prefix + ": '");
		int nulls = 0;
		int limit = buffer.limit();
		for (int i = 0; i < limit; i++) {
			char c = (char) buffer.get(i);
			if (c == '\u0000') {
				nulls++;
				continue;
			}
			if (nulls != 0) {
				System.out.print("|[" + nulls + " nulls]|");
				nulls = 0;
			}
			System.out.print(c);
		}
		System.out.println("'");
	}
}