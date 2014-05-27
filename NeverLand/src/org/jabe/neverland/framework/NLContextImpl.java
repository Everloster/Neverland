package org.jabe.neverland.framework;

import org.jabe.neverland.framework.http.AsyncHttpClient;

public class NLContextImpl extends NLContext {
	
	private NLContextImpl() {
		mAsyncHttpClient = new AsyncHttpClient();
	}
	
	private static volatile NLContextImpl instance;
	
	public static NLContextImpl getInstance() {
		synchronized (NLContextImpl.class) {
			if(instance == null) {
				instance = new NLContextImpl();
			}
			return instance;
		}
	}
	
	private AsyncHttpClient mAsyncHttpClient;
	
	@Override
	public Object getSystemService(String name) {
		if (name.equals(HTTP_CLIENT_SERVICE)) {
			return mAsyncHttpClient;
		}
		return null;
	}
	
}
