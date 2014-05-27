package org.jabe.neverland.view;

import org.jabe.neverland.R;
import org.jabe.neverland.util.SmileyParser;

import android.content.Context;
import android.util.AttributeSet;

public class DefaultFaceView extends FaceView {

    int mFaceSize = 0;
    int[] icons = SmileyParser.DEFAULT_SMILEY_RES_IDS;
    String[] names = getResources().getStringArray(SmileyParser.DEFAULT_SMILEY_NAMES);
    final int N = names.length;
    
    public DefaultFaceView(Context context) {
        super(context);
    }

    public DefaultFaceView(Context context, AttributeSet attrs) {
        super(context, attrs);
        setItemSize(15);
        mFaceSize = context.getResources().getDimensionPixelOffset(R.dimen.face_view_size);
        setLayoutParam(mFaceSize, mFaceSize);
        initFaceInfo();
    }

    public void initFaceInfo() {
        for (int i = 0; i < N; i++) {
            FaceInfo info = new FaceInfo();
            info.setName(names[i]);
            info.setIcon(icons[i]);
            mFaceInfos.add(info);
        }
    }
}
