/**
 * DataHelper.java
 * com.nearme.base.netty.common
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-5-7 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.netty.common;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;

import org.jboss.netty.buffer.ChannelBuffer;
import org.jboss.netty.channel.MessageEvent;
import org.jboss.netty.handler.codec.http.HttpMessage;
import org.jboss.netty.handler.codec.http.HttpRequest;

import com.google.protobuf.ByteString;
import com.nearme.base.netty.common.ProtobufWrapperProtocol.Header;
import com.nearme.base.netty.common.ProtobufWrapperProtocol.RequestData;
import com.nearme.base.netty.common.ProtobufWrapperProtocol.ResponseData;
import com.oppo.base.common.OConstants;
import com.oppo.base.common.StringUtil;

/**
 * ClassName:DataHelper <br>
 * Function: 数据操作辅助类 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2012-5-7  下午04:28:22
 */
public class DataHelper {
	/**
	 * 未知IP的定义
	 */
	public static final String UNKNOWN_IP = "unknown";

	private static final AtomicLong ID_GENERATOR = new AtomicLong(0);

	public static final byte[] EMPTY_DATA = new byte[0];

	/**
	 * 生成一个递增的id
	 * @param
	 * @return
	 */
	public static long getNewId() {
		return ID_GENERATOR.incrementAndGet();
	}

	/**
	 * 获取HttpResponse中的数据
	 * @param
	 * @return
	 */
	public static byte[] getBytesFromHttp(HttpMessage msg) {
		return getBytesFromChannelBuffer(msg.getContent());
	}

	/**
	 * 从ChannelBuffer获取byte[]数据
	 * @param
	 * @return
	 */
	public static byte[] getBytesFromChannelBuffer(ChannelBuffer cb) {
		byte[] data = null;

		int readableBytes = cb.readableBytes();
		if(cb.hasArray() && cb.arrayOffset() == 0 && readableBytes == cb.capacity()) {
			data = cb.array();
		} else {
			data = new byte[readableBytes];
			cb.getBytes(0, data, 0, readableBytes);
		}

		return data;
	}

	/**
	 * 拼接出向业务层请求的protobuf实体
	 * @param
	 * @return
	 */
	public static RequestData getRequestData(int requestType, HttpRequest request, MessageEvent e, int timeout) {
		//数据
		byte[] data = getBytesFromChannelBuffer(request.getContent());

		//IP
		String clientIp = getClientIP(request, e);

		return getRequestDataByHttpRequest(requestType, data, request, clientIp, timeout);
	}

	/**
	 * 拼接出向业务层请求的protobuf实体
	 * @param
	 * @return
	 */
	public static RequestData getRequestData(int requestType, byte[] data, HttpRequest request, MessageEvent e, int timeout) {
		//IP
		String clientIp = getClientIP(request, e);

		return getRequestDataByHttpRequest(requestType, data, request, clientIp, timeout);
	}

	/**
	 * 拼接出向业务层请求的protobuf实体
	 * @param requestType 请求类型，根据此类型查找对应的处理对象
	 * @param data 请求数据
	 * @param HttpRequest 转发客户端的头
	 * @param clientIp 客户端IP
	 * @param timeout 超时时间
	 * @return
	 */
	public static RequestData getRequestDataByHttpRequest(int requestType, byte[] data,
			HttpRequest request, String clientIp, int timeout) {
		return (RequestData)getRequestDataByHttpRequest(requestType, data, request, clientIp, timeout, true);
	}

	/**
	 * 拼接出向业务层请求的protobuf实体
	 * @param requestType 请求类型，根据此类型查找对应的处理对象
	 * @param data 请求数据
	 * @param HttpRequest 转发客户端的头
	 * @param clientIp 客户端IP
	 * @param timeout 超时时间
	 * @param build	  是否build数据
	 * @return
	 */
	public static Object getRequestDataByHttpRequest(int requestType, byte[] data,
			HttpRequest request, String clientIp, int timeout, boolean build) {
		RequestData.Builder builder = RequestData.newBuilder();
		builder.setRequestType(requestType);
		builder.setTimeout(timeout);
		builder.setId(getNewId());
		builder.setClientIp(clientIp);

		//数据
		if(null != data) {
			builder.setRequestData(ByteString.copyFrom(data));
		}

		Set<String> headerNameSet = request.getHeaderNames();
		if(null != headerNameSet) {
			Header.Builder headerBuilder = Header.newBuilder();
			for(String key : headerNameSet) {
				headerBuilder.setKey(key);
				headerBuilder.setValue(request.getHeader(key));
				builder.addHeader(headerBuilder.build());
			}
		}

		return build ? builder.build() : builder;
	}

	/**
	 * 拼接出向业务层请求的protobuf实体
	 * @param requestType 请求类型，根据此类型查找对应的处理对象
	 * @param data 请求数据
	 * @param headers 转发客户端的头
	 * @param clientIp 客户端IP
	 * @param timeout 超时时间
	 * @return
	 */
	public static RequestData getRequestData(int requestType, byte[] data,
			Map<String, String> headers, String clientIp, int timeout) {
		return (RequestData)getRequestData(requestType, data, headers, clientIp, timeout, true);
	}

	/**
	 * 拼接出向业务层请求的protobuf实体
	 * @param requestType 请求类型，根据此类型查找对应的处理对象
	 * @param data 请求数据
	 * @param headers 转发客户端的头
	 * @param clientIp 客户端IP
	 * @param timeout 超时时间
	 * @return
	 */
	public static Object getRequestData(int requestType, byte[] data,
			Map<String, String> headers, String clientIp, int timeout, boolean build) {
		RequestData.Builder builder = RequestData.newBuilder();
		builder.setRequestType(requestType);
		builder.setTimeout(timeout);
		builder.setId(getNewId());
		builder.setClientIp(clientIp);

		//数据
		if(null != data) {
			builder.setRequestData(ByteString.copyFrom(data));
		}

		//包头
		if(null != headers) {
			Header.Builder headerBuilder = Header.newBuilder();
			for(String key : headers.keySet()) {
				headerBuilder.setKey(key);
				headerBuilder.setValue(headers.get(key));
				builder.addHeader(headerBuilder.build());
			}
		}

		return build ? builder.build() : builder;
	}

	/**
	 *
	 * @param
	 * @return
	 */
	public static ResponseData getResponseData(RequestData request, byte[] responseData) {
		return getResponseData(request, request.getRequestType(), NettyConstants.RESPONSE_NORMAL, OConstants.EMPTY_STRING, responseData);
	}

	/**
	 *
	 * @param
	 * @return
	 */
	public static ResponseData getResponseData(RequestData request, int responseType, byte[] responseData) {
		return getResponseData(request, responseType, NettyConstants.RESPONSE_NORMAL, OConstants.EMPTY_STRING, responseData);
	}

	/**
	 *
	 * @param
	 * @return
	 */
	public static ResponseData getResponseData(RequestData request, int responseType, int responseCode, String responseMsg, byte[] responseData) {
		return (ResponseData)getResponseData(request, responseType, responseCode, responseMsg, responseData, true);
	}

	/**
	 *
	 * @param
	 * @return
	 */
	public static Object getResponseData(RequestData request, int responseType, int responseCode, String responseMsg, byte[] responseData,
			boolean build) {
		ResponseData.Builder builder = ResponseData.newBuilder();
		builder.setResponseType(responseType);
		builder.setTimeout(request.getTimeout());
		builder.setRequestId(request.getId());
		builder.setCode(responseCode);
		builder.setMessage(responseMsg);

		if(null != responseData) {
			builder.setResponseData(ByteString.copyFrom(responseData));
		}

		return build ? builder.build() : builder;
	}

	/**
	 *
	 * @param
	 * @return
	 */
	public static ResponseData getResponseData(RequestData request, ByteString responseData) {
		return getResponseData(request, request.getRequestType(), NettyConstants.RESPONSE_NORMAL, OConstants.EMPTY_STRING, responseData);
	}

	/**
	 *
	 * @param
	 * @return
	 */
	public static ResponseData getResponseData(RequestData request, int responseType, ByteString responseData) {
		return getResponseData(request, responseType, NettyConstants.RESPONSE_NORMAL, OConstants.EMPTY_STRING, responseData);
	}

	/**
	 *
	 * @param
	 * @return
	 */
	public static ResponseData getResponseData(RequestData request, int responseType, int responseCode, String responseMsg, ByteString responseData) {
		return (ResponseData)getResponseData(request, responseType, responseCode, responseMsg, responseData, true);
	}

	/**
	 *
	 * @param
	 * @return
	 */
	public static Object getResponseData(RequestData request, int responseType, int responseCode, String responseMsg,
			ByteString responseData, boolean build) {
		ResponseData.Builder builder = ResponseData.newBuilder();
		builder.setResponseType(responseType);
		builder.setTimeout(request.getTimeout());
		builder.setRequestId(request.getId());
		builder.setCode(responseCode);
		builder.setMessage(responseMsg);

		if(null != responseData) {
			builder.setResponseData(responseData);
		}

		return build ? builder.build() : builder;
	}

	public static Map<String, String> getHeader(HttpRequest request) {
		//客户端的头
		Map<String, String> headers = null;
		Set<String> headerNameSet = request.getHeaderNames();
		if(null != headerNameSet) {
			headers = new HashMap<String, String>(headerNameSet.size());
			for(String key : headerNameSet) {
				headers.put(key, request.getHeader(key));
			}
		}

		return headers;
	}

	/**
	 * 获取客户端IP
	 * @param request
	 * @return
	 */
	public static String getClientIP(HttpRequest request, MessageEvent e) {
		//如果有反向代理，则取头部x-forwarded-for中的值
		String ip = request.getHeader("x-forwarded-for");

		//无反向代理，则取代理IP
		if (StringUtil.isNullOrEmpty(ip, false) || ip.equalsIgnoreCase(UNKNOWN_IP)) {
			ip = request.getHeader("Proxy-Client-IP");
		} else {
			return getRealIp(ip, true);
		}

		//未获取到代理IP，则取
		if (StringUtil.isNullOrEmpty(ip, false) || ip.equalsIgnoreCase(UNKNOWN_IP)) {
			ip = request.getHeader("WL-Proxy-Client-IP");
		} else {
			return getRealIp(ip, false);
		}

		if (StringUtil.isNullOrEmpty(ip, false) || ip.equalsIgnoreCase(UNKNOWN_IP)) {
			ip = e.getRemoteAddress().toString();
		}

		return getRealIp(ip, false);
	}

	public static String getRealIp(String ip, boolean forwarded) {
		if (forwarded) {
			//使用代理时将逗号前的截取为最终ip
			int index = ip.indexOf(',');
			if (index == -1) {
				return ip;
			} else {
				return ip.substring(0, index);
			}
		} else {
			int index0 = ip.indexOf('/');
			int index1 = ip.indexOf(':', index0);
			if(index1 >= 0) {
				return ip.substring(index0 + 1, index1);
			} else {
				return ip.substring(index0 + 1);
			}
		}
	}

	public static void main(String[] args) {
		System.out.println(getRealIp("1823/asd:df", false));
		System.out.println(getRealIp("1823/asd", false));
		System.out.println(getRealIp("::ffff:202.120.2.30", false));
		System.out.println(getRealIp("1823asd", false));
		System.out.println(getRealIp("s/", false));
	}
}

