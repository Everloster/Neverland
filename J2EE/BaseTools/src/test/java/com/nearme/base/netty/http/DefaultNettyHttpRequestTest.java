/**
 * DefaultNettyHttpRequestTest.java
 * com.nearme.base.netty.http
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2013-5-8 		80036381
 *
 * Copyright (c) 2013 NearMe, All Rights Reserved.
*/

package com.nearme.base.netty.http;

import java.net.MalformedURLException;
import java.util.HashMap;
import java.util.Map;

import org.jboss.netty.handler.codec.http.HttpResponse;
import org.jboss.netty.util.HashedWheelTimer;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import com.nearme.base.concurrent.ExecutorManager;
import com.nearme.base.netty.common.DataHelper;

/**
 * ClassName:DefaultNettyHttpRequestTest <br>
 * Function: TODO ADD FUNCTION <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-5-8  上午9:22:59
 */
public class DefaultNettyHttpRequestTest {
	private DefaultNettyHttpRequest httpRequest;

	@Before
	public void init() {
		System.out.println("---init---");
		httpRequest = new DefaultNettyHttpRequest(
				ExecutorManager.newCachedThreadPool(), new HashedWheelTimer());
	}

	@After
	public void dispose() {
		httpRequest.close();

		ExecutorManager.shutdownAll();
		System.out.println("---shutdown---");
	}

	@Test
	public void testPost() {
		try {
			final long start = System.currentTimeMillis();
			Map<String, String> map = new HashMap<String, String>();
			map.put("Authorization", "OEPAUTH realm=\"OEP\",IMSI=\"2D28F2E4863B2E27C8DA708DEB1D0948\",appID=\"002734770859118531\",pubKey=\"82b1d8d3e0ed9df77ab4151e9cc7caeb\",netMode=\"WIFI\",packageName=\"com.oppo.nearmemusic\",version=\"1\"");

			httpRequest.getResponseByPost("http://218.200.230.142:85/opServer/1.0/search/music/listbychart", map, "<?xml version='1.0' encoding='UTF-8'?><request><chartCode>100</chartCode><pageNumber>1</pageNumber><numberPerPage>10</numberPerPage></request>".getBytes(), new HttpResponseListener() {
				@Override
				public void onSuccess(HttpResponse response) {
					System.out.println(response.getStatus() + "\t" + (System.currentTimeMillis() - start));
					byte[] data = DataHelper.getBytesFromChannelBuffer(response.getContent());
					System.out.println("len:" + data.length + "succ:\r\n" + new String(data));
				}

				@Override
				public void onError(HttpResponse response) {
					System.out.println(response.getStatus() + "\t" + (System.currentTimeMillis() - start));
					System.out.println("error:\r\n" + new String(DataHelper.getBytesFromChannelBuffer(response.getContent())));
				}
			}, 6000000);

			Thread.sleep(11000);
		} catch (MalformedURLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {

			// TODO Auto-generated catch block
			e.printStackTrace();

		}
	}

	@Test
	public void testPost1() {
		try {
			final long start = System.currentTimeMillis();
			Map<String, String> map = new HashMap<String, String>();
			map.put("Authorization", "OEPAUTH realm=\"OEP\",IMSI=\"2D28F2E4863B2E27C8DA708DEB1D0948\",appID=\"002734770859118531\",pubKey=\"82b1d8d3e0ed9df77ab4151e9cc7caeb\",netMode=\"WIFI\",packageName=\"com.oppo.nearmemusic\",version=\"1\"");

			httpRequest.getResponseByPost("http://localhost:8080/NearmeAppstore/user/user_login", map, "userName=youaremoon@126.com&password=ilveyou".getBytes(), new HttpResponseListener() {
				@Override
				public void onSuccess(HttpResponse response) {
					System.out.println(response.getStatus() + "\t" + (System.currentTimeMillis() - start));
					System.out.println("succ:\r\n" + new String(DataHelper.getBytesFromChannelBuffer(response.getContent())));
				}

				@Override
				public void onError(HttpResponse response) {
					System.out.println(response.getStatus() + "\t" + (System.currentTimeMillis() - start));
					System.out.println("error:\r\n" + new String(DataHelper.getBytesFromChannelBuffer(response.getContent())));
				}
			});

			Thread.sleep(11000);
		} catch (MalformedURLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {

			// TODO Auto-generated catch block
			e.printStackTrace();

		}
	}

	@Test
	public void testGetAndroidesk() {
		String url = "http://p.service.androidesk.com/cate/list";

		try {
			doGet(url, 2, 100);
			//Thread.currentThread().join();
		} catch (MalformedURLException e) {
			Assert.fail(e.getMessage());
		} catch (InterruptedException e) {
			Assert.fail(e.getMessage());
		}
	}

	@Test
	public void testBaiduSearch() {
		String url = "http://m.baidu.com/api?action=search&from=875i&token=oppo&type=oppo&word=qq&dpi=480_800&pn=0&rn=10";
		try {
			doGet(url, 2, 100);
			//Thread.currentThread().join();
		} catch (MalformedURLException e) {
			Assert.fail(e.getMessage());
		} catch (InterruptedException e) {
			Assert.fail(e.getMessage());
		}
	}

	@Test
	public void testGetNearme() {
		String url = "http://store.nearme.com.cn";

		try {
			doGet(url, 2, 100);
			//Thread.currentThread().join();
		} catch (MalformedURLException e) {
			Assert.fail(e.getMessage());
		} catch (InterruptedException e) {
			Assert.fail(e.getMessage());
		}
	}

	/**
	 * 进行get方式的request请求
	 * @param url	请求的url
	 * @param cnt	请求的次数
	 * @param sleepTimeInMills	两次请求间的间隔毫秒数，为0连续请求
	 * @return
	 * @throws InterruptedException
	 */
	public void doGet(final String url, int cnt, int sleepTimeInMills)
			throws MalformedURLException, InterruptedException {
		for(int i = 0; i < cnt; i++) {
			final long start = System.currentTimeMillis();
			httpRequest.getResponseByGet(url, null, new HttpResponseListener() {
				@Override
				public void onSuccess(HttpResponse response) {
					System.out.println(url + "\t" + response.getStatus() + "\t" + (System.currentTimeMillis() - start));
					System.out.println("succ:\r\n" + new String(DataHelper.getBytesFromChannelBuffer(response.getContent())));
				}

				@Override
				public void onError(HttpResponse response) {
					System.out.println(url + "\t" + response.getStatus() + "\t" + (System.currentTimeMillis() - start));
					System.out.println("error:\r\n" + new String(DataHelper.getBytesFromChannelBuffer(response.getContent())));
				}
			});

			if (sleepTimeInMills > 0 && i < cnt - 1) {
				Thread.sleep(sleepTimeInMills);
			}
		}

		Thread.sleep(10000);
	}
}

