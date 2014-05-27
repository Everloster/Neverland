package org.jabe.neverland.graphics;

import android.content.Context;
import android.opengl.GLSurfaceView;
import android.view.MotionEvent;

public class CircleSurfaceView extends GLSurfaceView {

	//声明场景渲染器
	private CircleRenderer circleRenderer;
	//纹理ID
	int textureId;
	
	public CircleSurfaceView(Context context) {
		super(context);
		
		//创建场景渲染器
		circleRenderer = new CircleRenderer(context);
		//为当前视图设置渲染器
		setRenderer(circleRenderer);
		//设置渲染器模式为主动渲染
		setRenderMode(GLSurfaceView.RENDERMODE_CONTINUOUSLY);
	}

	
	
	/**
	 * 触控事件回调函数
	 */
	@Override
	public boolean onTouchEvent(MotionEvent event) {
		//当前触控位置的X坐标
		float x = event.getX();
		//当前触控位置的Y坐标
		float y = event.getY();
		
		//判断触控事件并处理相关业务
		switch(event.getAction()){
		case MotionEvent.ACTION_MOVE:	//触控划动事件
			//计算在Y轴上移动距离
			float dy = y - DataConfig.mPreviousY;
			//计算在x轴上移动距离
			float dx = x - DataConfig.mPreviousX;
			//设置圆形沿x轴旋转角度
			circleRenderer.getCircle().mAngleX += dy * DataConfig.TOUCH_SCALE_FACTOR;
			//设置圆形沿z轴旋转角度
			circleRenderer.getCircle().mAngleZ += dx * DataConfig.TOUCH_SCALE_FACTOR;
			//重绘视图
			requestRender();
		}
		//记录触控坐标
		DataConfig.mPreviousX = x;
		DataConfig.mPreviousY = y;
		return true;
	}

}
