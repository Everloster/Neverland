package org.jabe.neverland.jedis;

public class JedisServer {

	private static JedisServer sInstance;
	
	private JedisServer() {
		
	}
	
	public static JedisServer getInstance() {
		synchronized (JedisServer.class) {
			if (sInstance != null) {
				sInstance = new JedisServer();
			}
			return sInstance;
		}
	}
	
	public void registerListCB(String listName, JedisCallback callback) {
		
	}
}
