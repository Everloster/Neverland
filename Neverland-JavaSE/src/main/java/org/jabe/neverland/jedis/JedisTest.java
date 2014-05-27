package org.jabe.neverland.jedis;

import redis.clients.jedis.Jedis;

public class JedisTest {
	
	public static void main(String[] args) {
		final Jedis jedis = new Jedis("localhost");
		jedis.set("key", "value");
		System.out.println(jedis.get("key"));
		
	}
	
}
