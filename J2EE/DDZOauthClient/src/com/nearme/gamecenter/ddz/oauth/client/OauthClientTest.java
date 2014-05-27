/**
 * Copyright (C) 2014, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	lailong
 * Since	2014年1月8日
 */
package com.nearme.gamecenter.ddz.oauth.client;

import java.io.IOException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.http.HttpEntity;
import org.apache.http.HttpResponse;
import org.apache.http.client.ClientProtocolException;
import org.apache.http.client.ResponseHandler;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import org.json.JSONObject;
import org.junit.Assert;
/**
 * 
 * @Author	LaiLong
 * @Since	2014年1月8日
 */
public class OauthClientTest extends Assert{
	
	private boolean useTest = true;
	
	private Object lock = new Object();
	
	private volatile static AtomicInteger count = new AtomicInteger(0);
	private volatile static AtomicInteger requestCount = new AtomicInteger(0);
	
	public static void main(String[] args) {
		final long start = System.currentTimeMillis();
		Executor exe = Executors.newCachedThreadPool();
		for (int i = 0; i < 2000; i++) {
//			try {
//				Thread.currentThread().sleep(100);
//			} catch (InterruptedException e) {
//				e.printStackTrace();
//			}
			exe.execute(new Runnable() {
				
				@Override
				public void run() {
					new OauthClientTest().request();
				}
			});
		}
		System.out.println("cast : " + (System.currentTimeMillis() - start));
	}

	/**
	 * 
	 */
	private void request() {
		CloseableHttpClient httpclient = HttpClients.createDefault();
        try {
        	String server = "";
            if (useTest) {
            	server = "http://127.0.0.1:8099/check_token";
            } else {
            	server = "http://121.52.231.201:8099/check_token";
            }
            final HttpPost httpPost = new HttpPost(server);
            final JSONObject jsonObject = new JSONObject();
            jsonObject.put("tokenKey", "c811efe99121324ec6db52c72d424670");
            jsonObject.put("tokenSecret", "0dc67787f99c9aefd8b2c639a8ef7e8f");
            jsonObject.put("fromPlatform", "oppo");
            jsonObject.put("deviceId", "hehe");
            jsonObject.put("id", "14446487");
            jsonObject.put("number", (requestCount.addAndGet(1)));
            httpPost.setEntity(new StringEntity(jsonObject.toString()));
            httpPost.setHeader("Content-Type", "application/x-www-form-urlencoded");
            // Create a custom response handler
            ResponseHandler<String> responseHandler = new ResponseHandler<String>() {

                public String handleResponse(
                        final HttpResponse response) throws ClientProtocolException, IOException {
                    int status = response.getStatusLine().getStatusCode();
                    if (status >= 200 && status < 300) {
                        HttpEntity entity = response.getEntity();
                        return entity != null ? EntityUtils.toString(entity) : null;
                    } else {
                        throw new ClientProtocolException("Unexpected response status: " + status);
                    }
                }

            };
            String responseBody = httpclient.execute(httpPost, responseHandler);
            synchronized (lock) {
                System.out.println((count.addAndGet(1)) + responseBody);
			}
        } catch (ClientProtocolException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
            try {
				httpclient.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
        }
	}
}
