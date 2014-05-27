/**
 * NettyHttpContext.java
 * com.nearme.base.netty.client
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2013-3-20 		80036381
 *
 * Copyright (c) 2013 NearMe, All Rights Reserved.
*/

package com.nearme.base.netty.client;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import org.jboss.netty.channel.MessageEvent;
import org.jboss.netty.handler.codec.http.HttpRequest;
import org.jboss.netty.handler.codec.http.HttpResponse;
import org.jboss.netty.util.Timeout;

/**
 * ClassName:NettyHttpContext <br>
 * Function: http信息打包 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-3-20  上午9:39:58
 */
public class NettyHttpContext<T> implements Serializable {
	/**
	 * @since Ver 1.1
	 */
	private static final long serialVersionUID = 1L;

	private long startTime;
	private HttpRequest request;
	private HttpResponse response;
	private MessageEvent messageEvent;
	private Timeout timeout;
	private Object attachment;
	private boolean requestComplete;
	//服务移除监听器
	private List<T> listeners;

	/**
	 * 添加服务返回监听器
	 * @param
	 * @return
	 */
	public void addServerResponseListener(T listener) {
		initListeners();
		listeners.add(listener);
	}

	/**
	 * 添加服务返回监听器
	 * @param
	 * @return
	 */
	public void addServerResponseListeners(List<T> listener) {
		initListeners();
		listeners.addAll(listener);
	}

	/**
	 * 移除服务返回监听器
	 * @param
	 * @return
	 */
	public void removeServerResponseListener(T listener) {
		if(null == listeners) {
			return;
		}

		listeners.remove(listener);
	}

	private void initListeners() {
		if(null == listeners) {
			listeners = new ArrayList<T>(1);
		}
	}

	public List<T> getListeners() {
		return listeners;
	}

	/**
	 * 获取startTime
	 * @return  the startTime
	 * @since   Ver 1.0
	 */
	public long getStartTime() {
		return startTime;
	}
	/**
	 * 设置startTime
	 * @param   startTime
	 * @since   Ver 1.0
	 */
	public void setStartTime(long startTime) {
		this.startTime = startTime;
	}
	/**
	 * 获取request
	 * @return  the request
	 * @since   Ver 1.0
	 */
	public HttpRequest getRequest() {
		return request;
	}
	/**
	 * 设置request
	 * @param   request
	 * @since   Ver 1.0
	 */
	public void setRequest(HttpRequest request) {
		this.request = request;
	}
	/**
	 * 获取response
	 * @return  the response
	 * @since   Ver 1.0
	 */
	public HttpResponse getResponse() {
		return response;
	}
	/**
	 * 设置response
	 * @param   response
	 * @since   Ver 1.0
	 */
	public void setResponse(HttpResponse response) {
		this.response = response;
	}
	/**
	 * 获取messageEvent
	 * @return  the messageEvent
	 * @since   Ver 1.0
	 */
	public MessageEvent getMessageEvent() {
		return messageEvent;
	}
	/**
	 * 设置messageEvent
	 * @param   messageEvent
	 * @since   Ver 1.0
	 */
	public void setMessageEvent(MessageEvent messageEvent) {
		this.messageEvent = messageEvent;
	}
	/**
	 * 获取attachment
	 * @return  the attachment
	 * @since   Ver 1.0
	 */
	public Object getAttachment() {
		return attachment;
	}
	/**
	 * 设置attachment
	 * @param   attachment
	 * @since   Ver 1.0
	 */
	public void setAttachment(Object attachment) {
		this.attachment = attachment;
	}
	/**
	 * 获取requestComplete
	 * @return  the requestComplete
	 * @since   Ver 1.0
	 */
	public boolean isRequestComplete() {
		return requestComplete;
	}
	/**
	 * 设置requestComplete
	 * @param   requestComplete
	 * @since   Ver 1.0
	 */
	public void setRequestComplete(boolean requestComplete) {
		this.requestComplete = requestComplete;
	}

	/**
	 * 获取timeout
	 * @return  the timeout
	 * @since   Ver 1.0
	 */
	public Timeout getTimeout() {
		return timeout;
	}

	/**
	 * 设置timeout
	 * @param   timeout
	 * @since   Ver 1.0
	 */
	public void setTimeout(Timeout timeout) {
		this.timeout = timeout;
	}
}

