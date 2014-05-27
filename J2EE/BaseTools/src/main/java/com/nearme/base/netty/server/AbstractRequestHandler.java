/**
 * AbstractRequestHandler.java
 * com.nearme.base.netty.server
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-1-29 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.netty.server;

import org.jboss.netty.channel.MessageEvent;
import org.jboss.netty.handler.codec.http.HttpMethod;
import org.jboss.netty.handler.codec.http.HttpRequest;
import org.jboss.netty.handler.codec.http.HttpResponse;
import org.jboss.netty.handler.codec.http.HttpResponseStatus;

import com.nearme.base.netty.common.RequestConfig;
import com.nearme.base.stat.ExecutorStat;
import com.nearme.base.stat.ExecutorStatPool;

/**
 * ClassName:AbstractRequestHandler <br>
 * Function: 抽象的IRequestHandler实现基本的处理逻辑METHOD_NOT_ALLOWED。<br>
 * 如果想只实现post方式，则重写doPost；如果想只实现get方式，则重写doGet方式；否则直接重写handleRequest方法即可。 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2012-1-29  下午02:52:40
 */
public abstract class AbstractRequestHandler implements IRequestHandler {

	@Override
	public void init(RequestConfig requestConfig) {
	}

	@Override
	public void handleRequest(HttpRequest request, HttpResponse response,
			MessageEvent e) {
		HttpMethod m = request.getMethod();
		if(HttpMethod.POST.equals(m)) {
			doPost(request, response, e);
		} else if(HttpMethod.GET.equals(m)) {
			doGet(request, response, e);
		}
	}

	/**
	 * POST方法处理
	 * @param
	 * @return
	 */
	protected void doPost(HttpRequest request, HttpResponse response,
			MessageEvent e) {
		methodNotAllowed(response, e);
	}

	/**
	 * GET方法处理
	 * @param
	 * @return
	 */
	protected void doGet(HttpRequest request, HttpResponse response,
			MessageEvent e) {
		methodNotAllowed(response, e);
	}

	/**
	 * 表示不支持指定方法的http输出
	 * @param
	 * @return
	 */
	protected void methodNotAllowed(HttpResponse response, MessageEvent e) {
		response.setStatus(HttpResponseStatus.METHOD_NOT_ALLOWED);

		output(response, e);
	}

	/**
	 * 设置字符编码为utf-8
	 * @param
	 * @return
	 */
	protected void setContentTypeToUtf8(HttpResponse response, MessageEvent e) {
		response.setHeader("Content-Type", "text/html; charset=UTF-8");
	}

	/**
	 * 设置页面没有缓存
	 *
	 * @param response
	 */
	public void setNoCache(HttpResponse response, MessageEvent e) {
		response.setHeader("Pragma", "no-cache");
		response.setHeader("Cache-Control", "no-cache");
	}

	/**
	 * 输出response中的内容但未关闭当前连接
	 * @param
	 * @return
	 */
	public void output(HttpResponse response, MessageEvent e) {
		NettyHttpOutput.output(response, e);
	}

	/**
	 * 输出指定的byte数组到response中并关闭当前连接
	 * @param data 输出的字节
	 * @param response
	 * @param e
	 * @return
	 */
	public void output(byte[] data, HttpResponse response, MessageEvent e) {
		NettyHttpOutput.output(data, response, e);
	}

	/**
	 * 添加统计信息
	 * @param isSuccess 是否执行成功
	 * @param executeTime 总执行时间
	 * @return
	 */
	protected void addStat(boolean isSuccess, long executeTime) {
		ExecutorStat stat = ExecutorStatPool.getInstance().getExecutorStat(this.getClass().getName());
		stat.addStat(isSuccess, executeTime);
	}

	/**
	 * 添加统计信息
	 * @param isSuccess 是否执行成功
	 * @param startTime 开始执行时间
	 * @return
	 */
	protected void addStatWithStartTime(boolean isSuccess, long startTime) {
		addStat(isSuccess, System.currentTimeMillis() - startTime);
	}
}

