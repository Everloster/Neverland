/**
 * ProtobufInOneEncoder.java
 * com.nearme.base.netty.codec
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-12-24 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.netty.codec;

import static org.jboss.netty.buffer.ChannelBuffers.wrappedBuffer;

import org.jboss.netty.buffer.ChannelBuffer;
import org.jboss.netty.buffer.ChannelBufferOutputStream;
import org.jboss.netty.channel.Channel;
import org.jboss.netty.channel.ChannelHandlerContext;
import org.jboss.netty.handler.codec.oneone.OneToOneEncoder;

import com.google.protobuf.CodedOutputStream;
import com.google.protobuf.MessageLite;

/**
 * ClassName:ProtobufInOneEncoder <br>
 * Function: pb类型数据编码,整合ProtobufVarint32LengthFieldPrepender和ProtobufEncoder <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-12-24  上午11:37:31
 */
public class ProtobufInOneEncoder extends OneToOneEncoder {
	
	public ProtobufInOneEncoder() {
		super();
	}
	
	@Override
    protected Object encode(
            ChannelHandlerContext ctx, Channel channel, Object msg0) throws Exception {
		Object msg = encodePb(ctx, channel, msg0);
		if (!(msg instanceof ChannelBuffer)) {
            return msg;
        }
        
        ChannelBuffer body = (ChannelBuffer) msg;
        int length = body.readableBytes();
        ChannelBuffer header =
            channel.getConfig().getBufferFactory().getBuffer(
                    body.order(),
                    CodedOutputStream.computeRawVarint32Size(length));
        CodedOutputStream codedOutputStream = CodedOutputStream
                .newInstance(new ChannelBufferOutputStream(header));
        codedOutputStream.writeRawVarint32(length);
        codedOutputStream.flush();
        return wrappedBuffer(header, body);
    }
	
	/**
	 * 将protobuf装换为ChannelBuffer
	 * @param 
	 * @return
	 */
	private Object encodePb(ChannelHandlerContext ctx, Channel channel, Object msg) throws Exception {
		if (msg instanceof MessageLite) {
            return wrappedBuffer(((MessageLite) msg).toByteArray());
        }
		
        if (msg instanceof MessageLite.Builder) {
            return wrappedBuffer(((MessageLite.Builder) msg).build().toByteArray());
        }
        
        return msg;
	}
}

