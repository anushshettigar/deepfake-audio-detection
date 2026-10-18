import { useEffect, useRef } from 'react'
import './Intro.css'

const word = 'DEEPSHIELD'

function Intro() {
  const sceneRef = useRef(null)
  const streaksRef = useRef(null)
  const particlesRef = useRef(null)
  const wordmarkRef = useRef(null)
  const extrusionRef = useRef(null)

  useEffect(() => {
    const wordmark = wordmarkRef.current
    const streaks = streaksRef.current
    const particles = particlesRef.current
    const extrusion = extrusionRef.current

    wordmark.innerHTML = ''
    ;[...word].forEach((letter, i) => {
      const span = document.createElement('span')
      span.className = 'ch'
      span.textContent = letter
      span.style.animationDelay = `${2.6 + i * 0.05}s`
      wordmark.appendChild(span)
    })

    streaks.innerHTML = ''
    for (let i = 0; i < 12; i++) {
      const streak = document.createElement('div')
      streak.className = 'streak'
      const rot = (360 / 12) * i + (Math.random() * 10 - 5)
      streak.style.setProperty('--rot', `${rot}deg`)
      streak.style.animationDelay = `${0.05 + Math.random() * 0.15}s`
      streaks.appendChild(streak)
    }

    particles.innerHTML = ''
    for (let i = 0; i < 20; i++) {
      const particle = document.createElement('div')
      particle.className = 'p'

      const z = -40 - Math.random() * 220
      const x0 = Math.random() * 420 - 210
      const y0 = Math.random() * 420 - 210
      const x1 = x0 + Math.random() * 60 - 30
      const y1 = y0 - 60 - Math.random() * 80

      particle.style.setProperty('--z', `${z}px`)
      particle.style.setProperty('--x0', `${x0}px`)
      particle.style.setProperty('--y0', `${y0}px`)
      particle.style.setProperty('--x1', `${x1}px`)
      particle.style.setProperty('--y1', `${y1}px`)
      particle.style.animationDuration = `${6 + Math.random() * 6}s`
      particle.style.animationDelay = `${Math.random() * 4}s`

      particles.appendChild(particle)
    }

    extrusion.innerHTML = ''

    const shieldD =
      'M150,18 L258,58 L258,172 C258,254 198,302 150,326 C102,302 42,254 42,172 L42,58 Z'

    for (let i = 1; i <= 6; i++) {
      const depth = i * 7
      const opacity = 0.3 - i * 0.04

      const wrap = document.createElement('div')
      wrap.className = 'shield-layer'
      wrap.style.transform = `translateZ(-${depth}px)`
      wrap.style.transformStyle = 'preserve-3d'

      const svg = document.createElementNS(
        'http://www.w3.org/2000/svg',
        'svg'
      )

      svg.setAttribute('viewBox', '0 0 300 340')
      svg.style.position = 'absolute'
      svg.style.inset = '0'
      svg.style.width = '100%'
      svg.style.height = '100%'

      const path = document.createElementNS(
        'http://www.w3.org/2000/svg',
        'path'
      )

      path.setAttribute('d', shieldD)
      path.style.stroke = `rgba(47,107,255,${Math.max(opacity, 0.03)})`
      path.style.animationDelay = `${0.9 + i * 0.03}s`

      svg.appendChild(path)
      wrap.appendChild(svg)
      extrusion.appendChild(wrap)
    }

    let targetX = 0
    let targetY = 0
    let currentX = 0
    let currentY = 0
    let animationFrame

    const handleMouseMove = (event) => {
      const px = event.clientX / window.innerWidth - 0.5
      const py = event.clientY / window.innerHeight - 0.5

      targetY = px * 14
      targetX = -py * 10
    }

    const animate = () => {
      currentX += (targetX - currentX) * 0.05
      currentY += (targetY - currentY) * 0.05

      sceneRef.current?.style.setProperty('--px', `${currentX}deg`)
      sceneRef.current?.style.setProperty('--py', `${currentY}deg`)

      animationFrame = requestAnimationFrame(animate)
    }

    window.addEventListener('mousemove', handleMouseMove)
    animationFrame = requestAnimationFrame(animate)

    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      cancelAnimationFrame(animationFrame)
    }
  }, [])

  const replay = () => {
    window.location.reload()
  }

  return (
    <div className="stage">
      <div className="grid" />

      <div className="scene-inner" ref={sceneRef}>
        <div className="orb-back" />
        <div className="floor-glow" />
        <div className="contact-shadow" />
        <div className="particles" ref={particlesRef} />
        <div className="streaks" ref={streaksRef} />

        <div className="shield-stage">
          <div ref={extrusionRef} />

          <svg className="shield" viewBox="0 0 300 340">
            <defs>
              <linearGradient id="strokeGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#00D4FF" />
                <stop offset="100%" stopColor="#2F6BFF" />
              </linearGradient>

              <linearGradient id="fillGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="rgba(0,212,255,0.16)" />
                <stop offset="100%" stopColor="rgba(47,107,255,0.03)" />
              </linearGradient>

              <clipPath id="shieldClip">
                <path d="M150,18 L258,58 L258,172 C258,254 198,302 150,326 C102,302 42,254 42,172 L42,58 Z" />
              </clipPath>

              <linearGradient id="scanGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="rgba(0,212,255,0)" />
                <stop offset="50%" stopColor="rgba(0,212,255,0.9)" />
                <stop offset="100%" stopColor="rgba(0,212,255,0)" />
              </linearGradient>
            </defs>

            <path
              className="shield-fill"
              d="M150,18 L258,58 L258,172 C258,254 198,302 150,326 C102,302 42,254 42,172 L42,58 Z"
            />

            <path
              className="shield-path"
              d="M150,18 L258,58 L258,172 C258,254 198,302 150,326 C102,302 42,254 42,172 L42,58 Z"
            />

            <path className="facet" d="M150,18 L150,326" />
            <path className="facet" d="M42,110 L258,110" />

            <g clipPath="url(#shieldClip)">
              <rect
                className="scan"
                x="0"
                y="0"
                width="300"
                height="60"
                fill="url(#scanGrad)"
              />
            </g>

            <g
              className="shield-mark"
              stroke="#00D4FF"
              strokeWidth="6"
              strokeLinecap="round"
              strokeLinejoin="round"
              fill="none"
            >
              <path d="M120,178 L143,203 L188,150" />
            </g>
          </svg>

          <div className="specular" />
          <div className="ring" />
        </div>

        <div className="wordmark-wrap">
          <div className="wordmark" ref={wordmarkRef} />
          <span className="underline" />
        </div>
      </div>

      <div className="vignette" />

      <button
        className="replay"
        onClick={replay}
        aria-label="Replay animation"
      >
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <polyline points="1 4 1 10 7 10" />
          <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
        </svg>
      </button>
    </div>
  )
}

export default Intro