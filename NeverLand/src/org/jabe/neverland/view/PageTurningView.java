package org.jabe.neverland.view;


import org.jabe.neverland.R;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PointF;
import android.graphics.Region;
import android.view.MotionEvent;
import android.view.View;


public class PageTurningView extends View
{
	String TAG="angang";
	//Region region;
	int cornerx,cornery;
	int WIDTH=800,HEIGHT=480;
	PointF touch;
	PointF bezierstart1,bezierturn1,beziermid1,bezierend1;
	PointF bezierstart2,bezierturn2,beziermid2,bezierend2;
	
	Path path0,path1;
	Bitmap bitmap1,bitmap2,bitmap3;
	
	float midx,midy;
	
	Bitmap bitmap;
	Canvas canvas;
	Paint bitmappaint;
	
	Paint paint;
	

	public PageTurningView(Context context) {
		super(context);
		
		// TODO Auto-generated constructor stub
		path0 = new Path();
		path1 = new Path();
		
		touch = new PointF(); // 触摸点
		bezierstart1 = new PointF(); // 贝塞尔曲线起始点
		bezierturn1 = new PointF(); // 
		beziermid1 = new PointF(); // 贝塞尔曲线中点
		bezierend1 = new PointF(); // 贝塞尔曲线结束点

		bezierstart2 = new PointF(); // 另一条贝塞尔曲线
		bezierturn2 = new PointF();
		beziermid2 = new PointF();
		bezierend2 = new PointF();

		// ---------------------------------------
		bitmap = Bitmap.createBitmap(800, 480, Bitmap.Config.ARGB_8888);
		canvas = new Canvas(bitmap);		//最后在ondraw中画的canvas
		bitmappaint = new Paint(Paint.DITHER_FLAG);

		bitmap1=BitmapFactory.decodeResource(getResources(), R.drawable.feite1);
		//bitmap2=BitmapFactory.decodeResource(getResources(), R.drawable.bm2);
		bitmap3=BitmapFactory.decodeResource(getResources(), R.drawable.feite2);		
	}
	

	void CompCorner(float x,float y)
	{
		if (x <= WIDTH / 2)
			cornerx = 0;
		else
			cornerx= WIDTH;
		if (y <= HEIGHT / 2)
			cornery = 0;
		else
			cornery = HEIGHT;
	}
	
	@Override
	public boolean onTouchEvent(MotionEvent event) {
		// TODO Auto-generated method stub
		if (event.getAction() == MotionEvent.ACTION_MOVE) {
			canvas.drawColor(0xFFAAAAAA);
			touch.x = event.getX();
			touch.y = event.getY();
			this.postInvalidate();
		}
		if (event.getAction() == MotionEvent.ACTION_DOWN) {
			canvas.drawColor(0xFFAAAAAA);
			touch.x = event.getX();
			touch.y = event.getY();
			CompCorner(touch.x, touch.y);
			this.postInvalidate();
		}
		if (event.getAction() == MotionEvent.ACTION_UP) {
			canvas.drawColor(0xFFAAAAAA);
			touch.x = cornerx;
			touch.y = cornery;
			this.postInvalidate();
		}
		// return super.onTouchEvent(event);
		return true;
	}
	
	public PointF CompCross(PointF P1, PointF P2, PointF P3, PointF P4) {
		PointF Cross = new PointF();
		// 二元函数一般式：y=ax+b
		float a1 = (P2.y - P1.y) / (P2.x - P1.x);
		float b1 = ((P1.x * P2.y) - (P2.x * P1.y)) / (P1.x - P2.x);

		float a2 = (P4.y - P3.y) / (P4.x - P3.x);
		float b2 = ((P3.x * P4.y) - (P4.x * P3.y)) / (P3.x - P4.x);
		Cross.x = (b2 - b1) / (a1 - a2);
		Cross.y = a1 * Cross.x + b1;
		return Cross;
	}

	
	private void CompPoints() {

		midx = (touch.x + cornerx) / 2;
		midy = (touch.y + cornery) / 2;

		bezierturn1.x = midx - (cornery - midy) * (cornery - midy) / (cornerx - midx);
		bezierturn1.y = cornery;
		bezierturn2.x = cornerx;
		bezierturn2.y = midy - (cornerx - midx) * (cornerx - midx) / (cornery- midy);

		bezierstart1.x = bezierturn1.x - (cornerx - bezierturn1.x) / 2;
		bezierstart1.y = cornery;

		bezierstart2.x = cornerx;
		bezierstart2.y = bezierturn2.y - (cornery - bezierturn2.y) / 2;

		bezierend1 = CompCross(touch, bezierturn1, bezierstart1,
				bezierstart2);
		bezierend2 = CompCross(touch, bezierturn2, bezierstart1,
				bezierstart2);
		beziermid1.x = (bezierstart1.x + 2 * bezierturn1.x + bezierend1.x) / 4;
		beziermid1.y = (2 * bezierturn1.y + bezierstart1.y + bezierend1.y) / 4;
		beziermid2.x = (bezierstart2.x + 2 * bezierturn2.x + bezierend2.x) / 4;
		beziermid2.y = (2 * bezierturn2.y + bezierstart2.y + bezierend2.y) / 4;

	}
	
	
	private void drawCurrentPageArea(Canvas canvas, Bitmap bitmap) {
		path0.reset();
		path0.moveTo(bezierstart1.x, bezierstart1.y);
		path0.quadTo(bezierturn1.x, bezierturn1.y, bezierend1.x, bezierend1.y);
		path0.lineTo(touch.x, touch.y);
		path0.lineTo(bezierend2.x, bezierend2.y);
		path0.quadTo(bezierturn2.x, bezierturn2.y, bezierstart2.x, bezierstart2.y);
		path0.lineTo(cornerx, cornery);
		path0.close();

		canvas.save();
		canvas.clipPath(path0, Region.Op.XOR);
		canvas.drawBitmap(bitmap, 0, 0, null);
		canvas.restore();
	}


	private void drawNextPageArea(Canvas canvas, Bitmap bitmap) {
		path1.reset();
		path1.moveTo(bezierstart1.x, bezierstart1.y);
		path1.lineTo(beziermid1.x, beziermid1.y);
		path1.lineTo(beziermid2.x, beziermid2.y);
		path1.lineTo(bezierstart2.x, bezierstart2.y);
		path1.lineTo(cornerx, cornery);
		path1.close();
	
		canvas.save();
		canvas.clipPath(path0);
		canvas.clipPath(path1, Region.Op.INTERSECT);
		canvas.drawBitmap(bitmap, 0, 0, null);

		canvas.restore();
	}
	
	@Override
	protected void onDraw(Canvas tempcanvas) {
		canvas.drawColor(0xFFAAAAAA);
		CompPoints();
		drawCurrentPageArea(canvas, bitmap1);		//第一页前面  先缓冲到canvas画布上
		drawNextPageArea(canvas, bitmap3);					//第二页前面  先缓冲到canvas画布上

		tempcanvas.drawBitmap(bitmap, 0, 0, bitmappaint);
	}
	

}
