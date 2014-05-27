package org.jabe.neverland.framework;

public abstract class NLContext {
	
	public static final String HTTP_CLIENT_SERVICE = "http_client_service";
	
	public abstract Object getSystemService(String name);
	
}
