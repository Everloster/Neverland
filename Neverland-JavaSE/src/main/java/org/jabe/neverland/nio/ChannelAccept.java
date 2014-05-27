package org.jabe.neverland.nio;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.ServerSocketChannel;
import java.nio.channels.SocketChannel;
import java.net.InetSocketAddress;

/**
 * Test nonblocking accept( ) using ServerSocketChannel. Start this program,
 * then "telnet localhost 1234" to connect to it.
 * 
 * @author Ron Hitchens (ron@ronsoft.com)
 */
public class ChannelAccept {
	public static final String GREETING = "Hello I must be going.\r\n";

	public static void main(String[] argv) throws Exception {
		int port = 12340; // default
		if (argv.length > 0) {
			port = Integer.parseInt(argv[0]);
		}
		ByteBuffer buffer = ByteBuffer.wrap(GREETING.getBytes());
		ServerSocketChannel ssc = ServerSocketChannel.open();
		ssc.socket().bind(new InetSocketAddress(port));
		ssc.configureBlocking(false);
		System.out.println("Waiting for connections");
		while (true) {
			SocketChannel sc = ssc.accept();
			if (sc == null) {
				// no connections, snooze a while
				Thread.sleep(2000);
			} else {
				System.out.println("Incoming connection from: "
						+ sc.socket().getRemoteSocketAddress());
				buffer.clear();
				sc.write(buffer);
				
				for (int i = 0; i < 5; i++) {
					try {
						Thread.sleep(1000);
						buffer.clear();
						if (sc.isConnected()) {
							sc.write(buffer);
						}
					} catch (IOException e) {
						// TODO: handle exception
						break;
					}
				}
				
				sc.close();
			}
		}
	}
}
