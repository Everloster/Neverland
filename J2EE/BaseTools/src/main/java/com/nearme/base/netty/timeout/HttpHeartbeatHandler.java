/**
 * HttpHeartbeatHandler.java
 * com.nearme.base.netty.server.timeout
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2013-4-11 		80036381
 *
 * Copyright (c) 2013 NearMe, All Rights Reserved.
*/

package com.nearme.base.netty.timeout;

import org.jboss.netty.channel.Channel;
import org.jboss.netty.channel.ChannelHandlerContext;
import org.jboss.netty.handler.codec.http.DefaultHttpResponse;
import org.jboss.netty.handler.codec.http.HttpResponseStatus;
import org.jboss.netty.handler.codec.http.HttpVersion;
import org.jboss.netty.handler.timeout.IdleState;
import org.jboss.netty.handler.timeout.IdleStateAwareChannelHandler;
import org.jboss.netty.handler.timeout.IdleStateEvent;

import com.nearme.base.netty.server.NettyHttpOutput;

/**
 * ClassName:HttpHeartbeatHandler <br>
 * Function: 心跳检测 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-4-11  下午9:10:54
 */
public class HttpHeartbeatHandler extends IdleStateAwareChannelHandler {
	/**
     * Invoked when a {@link Channel} has been idle for a while.
     */
    public void channelIdle(ChannelHandlerContext ctx, IdleStateEvent e) throws Exception {
    	if (e.getState() == IdleState.READER_IDLE) {
            //超过时间未读取到数据则关闭
    		e.getChannel().close();
        } else if (e.getState() == IdleState.WRITER_IDLE) {
            //超过时间未写数据发送心跳包检测
        	DefaultHttpResponse response = new DefaultHttpResponse(HttpVersion.HTTP_1_1, HttpResponseStatus.OK);
        	NettyHttpOutput.output(response, e.getChannel());
        } else if (e.getState() == IdleState.ALL_IDLE) {
            //不处理，由上面的读写状态监控
        }
    }
}

