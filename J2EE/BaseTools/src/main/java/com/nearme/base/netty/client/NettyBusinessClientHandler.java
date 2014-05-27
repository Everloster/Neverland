/**
 * NettyBusinessClientHandler.java
 * com.nearme.base.netty.client
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-5-4 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.netty.client;

import java.util.List;
import java.util.concurrent.Executor;

import org.jboss.netty.channel.ChannelHandlerContext;
import org.jboss.netty.channel.MessageEvent;

import com.nearme.base.netty.common.ProtobufWrapperProtocol.RequestData;
import com.nearme.base.netty.common.ProtobufWrapperProtocol.ResponseData;

/**
 * ClassName:NettyBusinessClientHandler <br>
 * Function: 客户端处理 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2012-5-4  下午05:25:42
 */
public class NettyBusinessClientHandler extends AbstractNettyBusinessClientHandler<RequestData, ResponseData> {
	private Executor executor;

//	public NettyBusinessClientHandler(
//			AbstractNettyBusinessClientFactory<RequestData, ResponseData> factory,
//			String identifier){
//		super(factory, identifier);
//	}

	public NettyBusinessClientHandler(
			AbstractNettyBusinessClientFactory<RequestData, ResponseData> factory,
			String identifier,
			Executor executor){
		super(factory, identifier);

		this.executor = executor;
	}

	public void messageReceived(ChannelHandlerContext ctx, MessageEvent e) throws Exception {
		Object obj = e.getMessage();
		if(obj instanceof ResponseData){
			putResponse((ResponseData)obj);
		} else if(obj instanceof List) {
			List<ResponseData> resList = (List<ResponseData>)obj;
			for(int i = 0, size = resList.size(); i < size; i++) {
				putResponse(resList.get(i));
			}
		} else{
			throw new Exception("仅支持指定的返回类型:ResponseData,List<ResponseData>:" + ((null == obj) ? "null" : obj.getClass().getSimpleName()));
		}
	}

	public void putResponse(final ResponseData responseData) throws Exception {
		final INettyClient<RequestData, ResponseData> client = getClient();
		if(null == executor) {
			client.putResponse(responseData);
		} else {
			//采取异步执行的方式
			executor.execute(new Runnable() {
				@Override
				public void run() {
					try {
						client.putResponse(responseData);
					} catch (Exception e) {
						error("处理返回数据失败", e);
					}
				}
			});
		}
	}
}
