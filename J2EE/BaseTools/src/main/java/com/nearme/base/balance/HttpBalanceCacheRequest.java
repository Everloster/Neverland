/**
 * HttpBalanceCacheRequest.java
 * com.nearme.base.balance
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2013-5-28 		80036381
 *
 * Copyright (c) 2013 NearMe, All Rights Reserved.
*/

package com.nearme.base.balance;

import java.io.File;
import java.io.InputStream;
import java.util.Map;

import com.nearme.base.balance.cache.DefaultBalanceCache;
import com.nearme.base.balance.cache.IBalanceCache;
import com.nearme.base.balance.model.ServerInfo;
import com.nearme.base.http.HttpConnectUtil;
import com.oppo.base.common.OConstants;
import com.oppo.base.common.StringUtil;

/**
 * ClassName:HttpBalanceCacheRequest <br>
 * Function: 按照负载配置进行分流的http请求，可以按照指定的缓存器进行缓存 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-5-28  上午9:23:09
 */
public class HttpBalanceCacheRequest extends HttpBalanceRequest {

	private IBalanceCache balanceCache;

	/**
	 * 文件方式的加载，此处默认为xml文件
	 *
	 * @param configFile
	 */
	public HttpBalanceCacheRequest(File configFile) {
		super(configFile);
	}

	/**
	 * InputStream方式的加载，此处默认为xml的InputStream
	 *
	 * @param inputStream
	 */
	public HttpBalanceCacheRequest(InputStream inputStream) {
		super(inputStream);
	}

	/**
	 * 使用get方式请求http数据
	 * @param url 请求的url
	 * @return
	 */
	public byte[] getResponseByGet(String groupId, String uri, Map<String, String> headers, String cacheIdentifier) throws Exception {
		ServerInfo server = this.getCacheConnectServer(groupId, cacheIdentifier);
		if(null == server) {
			return null;
		}

		String url = getRequestUrl(server.getServerUrl(), uri);
		try {
			byte[] data = HttpConnectUtil.getResponseByGet(url, headers);

			stat(groupId, server, null != data);
			return data;
		} catch(Exception ex) {
			stat(groupId, server, false);
			throw new Exception("url:" + url, ex);
		}
	}

	/**
	 * 使用post方式请求http数据
	 * @param urlLink 请求链接
	 * @param headers 请求头
	 * @param requestString 请求数据
	 * @return
	 */
	public byte[] getResponseByPost(String groupId, String uri, Map<String, String> headers, String requestString, String cacheIdentifier) throws Exception {
		return getResponseByPost(groupId, uri, headers, requestString.getBytes(OConstants.DEFAULT_ENCODING), cacheIdentifier);
	}

	/**
	 * 使用指定的方式请求http数据
	 * @param urlLink 请求链接
	 * @param headers 请求头
	 * @param requestString 请求数据
	 * @return
	 */
	public byte[] getResponseByPost(String groupId, String urlSuffix, Map<String, String> headers, byte[] requestBytes, String cacheIdentifier) throws Exception {
		ServerInfo server = this.getCacheConnectServer(groupId, cacheIdentifier);
		if(null == server) {
			return null;
		}

		String url = getRequestUrl(server.getServerUrl(), urlSuffix);

		try {
			byte[] data = HttpConnectUtil.getResponseByPost(url, headers, requestBytes);

			stat(groupId, server, null != data);
			return data;
		} catch(Exception ex) {
			stat(groupId, server, false);

			throw new Exception("url:" + url, ex);
		}
	}

	protected ServerInfo getCacheConnectServer(String groupId, String cacheIdentifier) throws Exception {
		IBalanceCache balanceCache = getBalanceCache();

		ServerInfo cacheServer = balanceCache.getCacheServer(groupId, cacheIdentifier);//缓存的ServerInfo
		ServerInfo connectServer = null;	//当前连接用的ServerInfo
		if (null != cacheServer && cacheServer.isActive()) {
			connectServer = cacheServer;
		}

		if (null == connectServer) {
			try {
				connectServer = balanceCache.getConnectServer(groupId, cacheIdentifier, super.getAllServers(groupId));
			} finally {
				// 加在finally块中保证在获取失败后将缓存设置为空
				balanceCache.putServerToCache(groupId, cacheIdentifier, connectServer);
			}
		}

		return connectServer;
	}

	/**
	 * 获取负载均衡的服务缓存
	 * @return  the balanceCache
	 * @since   Ver 1.0
	 */
	public IBalanceCache getBalanceCache() {
		if (null == balanceCache) {
			balanceCache = new DefaultBalanceCache();
		}

		return balanceCache;
	}

	/**
	 * 设置负载均衡的服务缓存
	 * @param   balanceCache
	 * @since   Ver 1.0
	 */
	public void setBalanceCache(IBalanceCache balanceCache) {
		this.balanceCache = balanceCache;
	}

	public static void main(String[] args) throws Exception {
		HttpBalanceCacheRequest hbcq = new HttpBalanceCacheRequest(new File("G:\\App_Store\\Code\\NewMarket\\Trunk\\Business\\java\\NearmeAppstoreCore\\src\\config\\external\\service-group.xml"));

		for (int i = 0; i < 1003; i++) {
			hbcq.getCacheConnectServer("home-recommend", "asdasd" + StringUtil.padLeft(String.valueOf(i), '0', 3));
			//System.out.println(s.getServerId());
		}
	}
}

